# ✅ 최적 실험 조합을 반영한 학습 코드 (2025-06)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

# ---------------- Dataset ----------------
class PostureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["class_id"]
        return image, label

# ---------------- 모델 구성 ----------------
def get_backbone():
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    in_features = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, in_features

def get_mlp_head(mlp_type, in_features):
    if mlp_type == "strong":
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    raise ValueError("Only 'strong' MLP is defined in this script.")

# ---------------- Epoch 학습 함수 ----------------
def train_epoch(loader, model, criterion, optimizer, is_train=True, threshold=0.6):
    model.train() if is_train else model.eval()
    preds, labels = [], []
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds_batch = (torch.sigmoid(outputs).detach().cpu().numpy() > threshold).astype(int)
        labels_batch = targets.cpu().numpy().astype(int)

        preds.extend(preds_batch)
        labels.extend(labels_batch)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return running_loss / len(loader.dataset), acc, f1

# ---------------- 전체 학습 루프 ----------------
def train_with_best_saving(model, train_loader, val_loader, criterion, optimizer,
                           threshold=0.6, num_epochs=15):
    best_score = -float('inf')
    loss_history = []
    save_path = f"best_model_f1loss_th{threshold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_epoch(train_loader, model, criterion, optimizer, True, threshold)
        val_loss, val_acc, val_f1 = train_epoch(val_loader, model, criterion, optimizer, False, threshold)

        loss_history.append(val_loss)
        min_l, max_l = min(loss_history), max(loss_history)
        norm_loss = (val_loss - min_l) / (max_l - min_l + 1e-8)
        score = val_f1 - norm_loss

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train F1: {train_f1:.3f}, Val F1: {val_f1:.3f}, "
              f"Val Loss: {val_loss:.4f}, Score(F1-normLoss): {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch+1} (F1={val_f1:.4f}, Loss={val_loss:.4f})")

    print(f"\n🎯 최종 저장 모델: {save_path}")
    return save_path

# ---------------- 실행 ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv("../dataset-modification/train_pose_parsed.csv")[["filename", "class_id"]]
    val_df = pd.read_csv("../dataset-modification/valid_pose_parsed.csv")[["filename", "class_id"]]
    train_df = train_df.groupby("filename")["class_id"].min().reset_index()
    val_df = val_df.groupby("filename")["class_id"].min().reset_index()

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df["class_id"]), y=train_df["class_id"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PostureDataset(train_df, "../dataset-modification/train-visualized/images", transform)
    val_dataset = PostureDataset(val_df, "../dataset-modification/valid-visualized/images", transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    backbone, in_features = get_backbone()
    mlp_head = get_mlp_head("strong", in_features)
    model = nn.Sequential(backbone, mlp_head).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    train_with_best_saving(model, train_loader, val_loader, criterion, optimizer,
                           threshold=0.4, num_epochs=15)
