# %%
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import glob

# %%
# 사용자 정의 Dataset 클래스
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

# %%
# EarlyStopping 클래스
class EarlyStopping:
    def __init__(self, patience=30, delta=0.0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"🟡 EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                print("🛑 EarlyStopping triggered! Stopping training.")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Validation loss가 개선될 때만 모델 저장"""
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"✅ Model saved to {self.checkpoint_path}")

# %%
# 백본 모델 로드 및 분류기 제거 함수
def get_backbone(name):
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
    elif name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    return model, in_features
# MLP Head 정의 함수
def get_mlp_head(in_features):
    return  nn.Linear(in_features, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 64)

# EarlyStopping 클래스
class EarlyStopping:
    def __init__(self, patience=30, delta=0.0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"🟡 EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                print("🛑 EarlyStopping triggered! Stopping training.")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Validation loss가 개선될 때만 모델 저장"""
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"✅ Model saved to {self.checkpoint_path}")

# 백본 모델 로드 및 분류기 제거 함수
def get_backbone(name):
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
    elif name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    return model, in_features

# MLP Head 정의 함수
def get_mlp_head(in_features):
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

# 에폭별 특성 데이터 저장 함수 (날짜 폴더 포함)
def save_epoch_data(epoch, features, labels, setname, backbone, save_dir='embeddings'):
    today = datetime.now().strftime('%Y%m%d')
    save_path = os.path.join(save_dir, today, backbone)
    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/epoch_{epoch:03d}_{setname}_features.npy", features)
    np.save(f"{save_path}/epoch_{epoch:03d}_{setname}_labels.npy", labels)

# 최신 날짜 폴더를 찾는 헬퍼 함수
def get_latest_date_folder(base_dir):
    date_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit() and len(f) == 8], reverse=True)
    if not date_folders:
        raise FileNotFoundError(f"No date-stamped folders found in {base_dir}")
    return date_folders[0]

# T-SNE 시각화 배치 처리 함수
def batch_visualize_tsne(backbone, embedding_dir='embeddings', save_dir='logs'):
    try:
        latest_date_folder = get_latest_date_folder(embedding_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    path = os.path.join(embedding_dir, latest_date_folder, backbone)
    if not os.path.exists(path):
        print(f"No embeddings found for {backbone} in {path}")
        return

    files = sorted([f for f in os.listdir(path) if f.endswith("_features.npy")])
    if not files:
        print(f"No feature files found for {backbone} in {path}")
        return

    for file in files:
        epoch = int(file.split('_')[1])
        features = np.load(os.path.join(path, file))
        labels = np.load(os.path.join(path, file.replace("features.npy", "labels.npy")))

        reduced = PCA(n_components=min(50, features.shape[1]), random_state=42).fit_transform(features)
        embedded = TSNE(n_components=2, random_state=42, perplexity=min(30, len(reduced) - 1)).fit_transform(reduced)

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f"{backbone} - Epoch {epoch:03d}")

        out_dir = os.path.join(save_dir, latest_date_folder, backbone)
        os.makedirs(out_dir, exist_ok=True)
        # JPG 포맷으로 저장
        plt.savefig(f"{out_dir}/epoch_{epoch:03d}.jpg")
        plt.close()

# 훈련/검증 데이터 T-SNE 시각화 함수
def visualize_tsne_train_val(epoch, backbone, embedding_dir='embeddings', save_dir='logs'):
    try:
        latest_date_folder = get_latest_date_folder(embedding_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    def load_and_embed(path_feat, path_label):
        features = np.load(path_feat)
        labels = np.load(path_label)
        reduced = PCA(n_components=min(50, features.shape[1]), random_state=42).fit_transform(features)
        embedded = TSNE(n_components=2, random_state=42, perplexity=min(30, len(reduced) - 1)).fit_transform(reduced)
        return embedded, labels

    base = os.path.join(embedding_dir, latest_date_folder, backbone)
    feat_train = os.path.join(base, f"epoch_{epoch:03d}_train_features.npy")
    lbls_train = os.path.join(base, f"epoch_{epoch:03d}_train_labels.npy")
    feat_val = os.path.join(base, f"epoch_{epoch:03d}_val_features.npy")
    lbls_val = os.path.join(base, f"epoch_{epoch:03d}_val_labels.npy")

    if not (os.path.exists(feat_train) and os.path.exists(feat_val)):
        print(f"Features for epoch {epoch:03d} not found for {backbone} in {base}. Skipping visualization.")
        return

    emb_train, y_train = load_and_embed(feat_train, lbls_train)
    emb_val, y_val = load_and_embed(feat_val, lbls_val)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].scatter(emb_train[:, 0], emb_train[:, 1], c=y_train, cmap='tab10', alpha=0.7)
    axs[0].set_title(f"Train - Epoch {epoch:03d}")

    axs[1].scatter(emb_val[:, 0], emb_val[:, 1], c=y_val, cmap='tab10', alpha=0.7)
    axs[1].set_title(f"Validation - Epoch {epoch:03d}")

    out_dir = os.path.join(save_dir, latest_date_folder, backbone)
    os.makedirs(out_dir, exist_ok=True)
    # JPG 포맷으로 저장
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}_trainval.jpg")
    plt.close()

# T-SNE 훈련/검증 시각화 일괄 처리 함수
def batch_visualize_tsne_trainval(backbone, embedding_dir='embeddings', save_dir='logs'):
    try:
        latest_date_folder = get_latest_date_folder(embedding_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    path = os.path.join(embedding_dir, latest_date_folder, backbone)
    if not os.path.exists(path):
        print(f"No embeddings found for {backbone} in {path}")
        return

    files = sorted([f for f in os.listdir(path) if f.endswith("_train_features.npy")])
    if not files:
        print(f"No train feature files found for {backbone} in {path}")
        return

    epochs = [int(f.split('_')[1]) for f in files]

    for epoch in epochs:
        visualize_tsne_train_val(epoch, backbone, embedding_dir, save_dir)

# 학습 에폭 함수
def train_epoch(loader, model, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    running_loss = 0.0
    preds, labels = [], []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.float().to(device).unsqueeze(1)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds_batch = (torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(int)
        targets_batch = targets.detach().cpu().numpy().astype(int)

        preds.extend(preds_batch)
        labels.extend(targets_batch)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return running_loss / len(loader.dataset), acc, f1




# %%
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# %%
# CSV 파일 불러오기
train_df = pd.read_csv("../dataset-modification/train_pose_parsed.csv")[["filename", "class_id"]]
valid_df = pd.read_csv("../dataset-modification/valid_pose_parsed.csv")[["filename", "class_id"]]

# filename 중복 처리 (class_id의 min 값 선택)
train_df = train_df.groupby("filename")["class_id"].min().reset_index()
valid_df = valid_df.groupby("filename")["class_id"].min().reset_index()

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(train_df["class_id"]), y=train_df["class_id"])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 이미지 경로 정의
train_image_dir = "../dataset-modification/train-visualized/images/"
valid_image_dir = "../dataset-modification/valid-visualized/images/"

# 데이터 증강 및 정규화 Transform 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 이미지 랜덤 크롭 후 리사이즈, 다양성 증가
#     transforms.RandomHorizontalFlip(p=0.5),              # 좌우 반전 유지
#     transforms.RandomVerticalFlip(p=0.2),                # 상하 반전 추가 (데이터 특성에 따라)
#     transforms.RandomRotation(degrees=30),               # 회전 각도 증가 (15 -> 30)
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # 색상 변화 강도 증가
#     transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1), shear=10), # 이동, 스케일, 전단 강도 증가
#     # transforms.ToTensor()는 Normalize 앞에 와야 합니다.
#     transforms.ToTensor(),
#     # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)), # 선택 사항: 이미지 일부를 랜덤하게 가림
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Dataset 생성
train_dataset = PostureDataset(train_df, train_image_dir, transform=train_transform)
val_dataset = PostureDataset(valid_df, valid_image_dir, transform=val_transform)

# # DataLoader 생성 (배치 크기 32)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count() // 2 or 1) # num_workers 추가
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count() // 2 or 1) # num_workers 추가
# # ✅ Jupyter용 설정
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


print("Train 클래스 분포:\n", train_df["class_id"].value_counts())
print("Valid 클래스 분포:\n", valid_df["class_id"].value_counts())
print("Class weights:", class_weights_tensor)

# %%
original_train_df = pd.read_csv("../dataset-modification/train_pose_parsed.csv")
print("원본 row 수:", len(original_train_df))
print("고유 filename 수:", original_train_df["filename"].nunique())

# %%
# 장치 설정
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 설정
backbone_name = "mobilenet_v3_large"
num_epochs = 40
log_interval = 1
checkpoint_path = f"{backbone_name}_best_model.pt"
results = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "train_f1": [], "val_f1": []
}

# 모델 구성
backbone, in_features = get_backbone(backbone_name)
mlp_head = get_mlp_head(in_features)
model = nn.Sequential(backbone, mlp_head).to(device)

# 손실 함수 및 옵티마이저
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1].to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# EarlyStopping
early_stopper = EarlyStopping(patience=15, delta=0.001, checkpoint_path=checkpoint_path)

# TensorBoard Writer
writer = SummaryWriter(log_dir=f"runs/{backbone_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드 설정
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

# 기존 PostureDataset, EarlyStopping, get_backbone, get_mlp_head, save_epoch_data 정의 유지

def train_epoch(loader, model, criterion, optimizer, device, is_train=True):
    print(f"Starting train_epoch, is_train={is_train}")
    model.train() if is_train else model.eval()
    running_loss = 0.0
    preds, labels = [], []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.float().to(device).unsqueeze(1)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds_batch = (torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(int)
        targets_batch = targets.detach().cpu().numpy().astype(int)

        preds.extend(preds_batch)
        labels.extend(targets_batch)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return running_loss / len(loader.dataset), acc, f1

def plot_training_metrics(results, save_path="training_metrics.jpg"):
    print(f"Plotting training metrics to {save_path}")
    epochs = range(1, len(results["train_loss"]) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, results["train_loss"], label="Train")
    plt.plot(epochs, results["val_loss"], '--', label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, results["train_acc"], label="Train")
    plt.plot(epochs, results["val_acc"], '--', label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, results["train_f1"], label="Train")
    plt.plot(epochs, results["val_f1"], '--', label="Val")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count() // 2)

    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc, train_f1 = train_epoch(train_loader, model, criterion, optimizer, device, is_train=True)
        val_loss, val_acc, val_f1 = train_epoch(val_loader, model, criterion, optimizer, device, is_train=False)

        print(f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)
        results["train_f1"].append(train_f1)
        results["val_f1"].append(val_f1)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break

        if epoch % log_interval == 0:
            model.eval()
            for split_name, loader in zip(["train", "val"], [train_loader, val_loader]):
                feats_list, labels_list = [], []
                with torch.no_grad():
                    for imgs, lbls in loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        feats = backbone(imgs)
                        feats_list.append(feats.cpu())
                        labels_list.append(lbls.cpu())
                all_feats = torch.cat(feats_list, dim=0).numpy()
                all_lbls = torch.cat(labels_list, dim=0).numpy()
                save_epoch_data(epoch, all_feats, all_lbls, split_name, backbone_name)

if __name__ == "__main__":
    main()
    plot_training_metrics(results)  # 훈련 완료 후 플롯
    # batch_visualize_tsne_trainval(backbone_name)  # 필요 시 주석 해제


