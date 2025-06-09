from ultralytics import YOLO
import cv2

# YOLOv8 segmentation 모델 로드
model = YOLO("yolov8x-seg.pt")

# 클래스 이름 목록
CLASS_NAMES = model.names
TARGET_CLASSES = {"person"}  # book은 COCO에 없음

# 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론 (Segmentation 포함)
    results = model(frame, verbose=False)[0]

    # 결과 프레임 복사
    output = frame.copy()

    for i in range(len(results.boxes)):
        cls_id = int(results.boxes.cls[i])
        cls_name = CLASS_NAMES[cls_id]
        score = float(results.boxes.conf[i])

        if cls_name not in TARGET_CLASSES:
            continue

        # 마스크 처리
        if results.masks is not None:
            mask = results.masks.data[i].cpu().numpy()
            mask = (mask * 255).astype("uint8")
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        # 바운딩 박스 및 클래스 표시
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[i].cpu().numpy())
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output, f"{cls_name} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("YOLOv8 Instance Segmentation", output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap.release()
    cv2.destroyAllWindows()
