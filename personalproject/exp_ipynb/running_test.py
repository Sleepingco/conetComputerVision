import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO

# ---------------- 모델 구성 (백본 + MLP Head) ----------------
def build_model(backbone_name="mobilenet_v3_large", mlp_type="simple"):
    # 백본 로딩
    if backbone_name == 'mobilenet_v3_large':
        backbone = models.mobilenet_v3_large(pretrained=False)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
    else:
        raise NotImplementedError

    # MLP Head 구성
    if mlp_type == "simple":
       head = nn.Sequential(
            nn.BatchNorm1d(in_features),     # 1.0.*
            nn.Linear(in_features, 64),      # 1.1.*
            nn.BatchNorm1d(64),              # 1.2.*
            nn.ReLU(),                       # 1.3
            nn.Dropout(0.5),                 # 1.4
            nn.Linear(64, 1)                 # 1.5
        )

    else:
        raise NotImplementedError

    return nn.Sequential(backbone, head)

# ---------------- 모델 로딩 ----------------
model = build_model("mobilenet_v3_large", "simple")
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

# ---------------- 전처리 ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- YOLO Pose 모델 로딩 ----------------
pose_model = YOLO("yolov8n-pose.pt")

# ---------------- 비디오 입력 ----------------
cap = cv2.VideoCapture("test_video.mp4")  # 웹캠 사용시 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # keypoint 예측
    results = pose_model(frame)
    keypoints_list = results[0].keypoints.xy.cpu().numpy()

    for kp in keypoints_list:
        # keypoint 시각화 (프레임에 바로 그리기)
        for x, y in kp:
            if not np.isnan([x, y]).any():
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 프레임 시각화 결과로 예측
        vis_frame = frame.copy()
        input_img = transform(vis_frame).unsqueeze(0)

        with torch.no_grad():
            output = model(input_img)
            pred = (torch.sigmoid(output) > 0.5).item()
            label = "Good" if pred else "Bad"

        x, y = int(kp[0][0]), int(kp[0][1])
        color = (0, 255, 0) if label == "Good" else (0, 0, 255)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.circle(frame, (x, y), 5, color, -1)

    cv2.imshow("Posture Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
