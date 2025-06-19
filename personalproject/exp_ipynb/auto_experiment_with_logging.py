import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from itertools import product
from datetime import datetime
import random

# ---------------- 기본 설정 ----------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset 정의 ----------------
class PostureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")
        label = row["class_id"]
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------- 모델 구성 ----------------
def get_backbone(name="mobilenet_v3_large"):
    model = models.mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, in_features

def get_mlp_head(name, in_features):
    if name == "mini":
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 1)
        )
    elif name == "simple":
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    elif name == "strong":
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

def get_augment(level):
    if level == "none":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif level == "medium":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 이미지 랜덤 크롭 후 224x224로 리사이즈 (다양한 스케일 학습)
            transforms.RandomHorizontalFlip(p=0.5),             # 50% 확률로 좌우 반전
            transforms.RandomVerticalFlip(p=0.2),               # 20% 확률로 상하 반전 (데이터 특성에 따라 조절 필요)
            transforms.RandomRotation(degrees=30),              # -30도에서 +30도 사이로 랜덤 회전
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # 색상, 밝기, 대비 등 랜덤 변경
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=10), # 이동, 스케일, 전단 변형
            # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)), # 선택 사항: 이미지 일부를 랜덤하게 가림 (강력한 증강)
            transforms.ToTensor(),                              # PIL Image를 PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 통계 기반 정규화
        ])

# ---------------- 학습/검증 루프 ----------------
def train_epoch(loader, model, criterion, optimizer, is_train=True):
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

        preds_batch = (torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(int)
        labels_batch = targets.cpu().numpy().astype(int)

        preds.extend(preds_batch)
        labels.extend(labels_batch)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return running_loss / len(loader.dataset), acc, f1

# ---------------- 실험 실행 ----------------
def run_experiment(mlp_type, aug_level, use_scheduler, batch_size):
    print(f"\n🔧 Experiment: MLP={mlp_type}, AUG={aug_level}, SCH={use_scheduler}, BS={batch_size}")
    
    # 데이터셋 로딩
    train_df = pd.read_csv("../dataset-modification/train_pose_parsed.csv")[["filename", "class_id"]]
    val_df = pd.read_csv("../dataset-modification/valid_pose_parsed.csv")[["filename", "class_id"]]
    train_df = train_df.groupby("filename")["class_id"].min().reset_index()
    val_df = val_df.groupby("filename")["class_id"].min().reset_index()

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df["class_id"]), y=train_df["class_id"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    train_transform = get_augment(aug_level)
    val_transform = get_augment("none")

    train_dataset = PostureDataset(train_df, "../dataset-modification/train-visualized/images", train_transform)
    val_dataset = PostureDataset(val_df, "../dataset-modification/valid-visualized/images", val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 모델 구성
    backbone, in_features = get_backbone()
    mlp_head = get_mlp_head(mlp_type, in_features)
    model = nn.Sequential(backbone, mlp_head).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) if use_scheduler else None

    for epoch in range(15):
        train_loss, train_acc, train_f1 = train_epoch(train_loader, model, criterion, optimizer, True)
        val_loss, val_acc, val_f1 = train_epoch(val_loader, model, criterion, optimizer, False)
        if scheduler:
            scheduler.step()

        print(f"[{epoch+1}/15] Train F1: {train_f1:.3f} | Val F1: {val_f1:.3f}")

# ---------------- 실험 반복 ----------------
if __name__ == "__main__":
    mlp_types = ['mini', 'simple', 'default', 'strong']
    augment_levels = ['none', 'medium', 'strong']
    use_schedulers = [False, True]
    batch_sizes = [16, 32]

    for mlp, aug, sch, bs in product(mlp_types, augment_levels, use_schedulers, batch_sizes):
        run_experiment(mlp, aug, sch, bs)
