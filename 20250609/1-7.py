from ultralytics import YOLO
import cv2

model = YOLO('yolov8x-seg.pt')

img = cv2.imread('busy_street.jpg')
result = model(img)

result[0].show()
