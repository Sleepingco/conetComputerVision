import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO

# ---------------- 모델 구성 (백본 + MLP Head) ----------------
def build_model(backbone_name="mobilenet_v3_large", mlp_type="simple"):
    if backbone_name == 'mobilenet_v3_large':
        backbone = models.mobilenet_v3_large(pretrained=False)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
    else:
        raise NotImplementedError

    if mlp_type == "simple":
        head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    else:
        raise NotImplementedError

    return nn.Sequential(backbone, head)

# ---------------- 모델 로딩 ----------------
model = build_model("mobilenet_v3_large", "simple")
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.load_state_dict(torch.load("best_model_f1loss_th0.6_20250620_104106.pt", map_location="cpu"))
# model.load_state_dict(torch.load("best_model_f1loss_th0.7_20250620_110031.pt", map_location="cpu"))
# model.load_state_dict(torch.load("best_model_f1loss_th0.4_20250620_112041.pt", map_location="cpu"))
model.eval()

# ---------------- 전처리 ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- 앉은 자세 판단 함수 ----------------
def is_sitting(keypoints):
    try:
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]

        if np.any(np.isnan([left_hip, right_hip, left_knee, right_knee])):
            return False

        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2

        return (knee_y - hip_y) < 40  # 무릎이 올라와 있으면 앉은 자세
    except:
        return False

# ---------------- YOLO Pose 모델 로딩 ----------------
pose_model = YOLO("yolov8n-pose.pt")

# ---------------- 비디오 입력 ----------------
cap = cv2.VideoCapture("test_video.mp4")  # 웹캠은 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame)
    keypoints_list = results[0].keypoints.xy.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for kp, box in zip(keypoints_list, boxes):
        if not is_sitting(kp):
            continue

        # keypoint 시각화
        for x, y in kp:
            if not np.isnan([x, y]).any():
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # bbox 영역 crop → 분류 모델 입력
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        input_img = transform(person_crop).unsqueeze(0)

        with torch.no_grad():
            output = model(input_img)
            pred = (torch.sigmoid(output) > 0.5).item()
            label = "Good" if pred else "Bad"

        # 시각화
        x, y = int(kp[0][0]), int(kp[0][1])  # 코 위치 기준
        color = (0, 255, 0) if label == "Good" else (0, 0, 255)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.circle(frame, (x, y), 5, color, -1)

    cv2.imshow("Posture Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
