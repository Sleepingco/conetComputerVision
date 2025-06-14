# -*- coding: utf-8 -*-
"""IoU 계산.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XMs7MSEdR6pST6eVA-C-iykAlYymSwLK
"""

# 두개의 박스가 서로 교차하는지를 판단하는 함수
def _is_box_intersect(box1,box2):
  if(
      abs(box1[0]-box2[0]) < box1[2]/2 + box2[2]/2
      and abs(box1[1] - box2[1]) < box1[3]/2 + box2[3]/2
  ):
    return True
  else:
    return False
# 박스의 영역을 계산하는 박스 box의 좌표체계는 [center_x,center_t,width,height]이고, 상대 좌표(YOLO 라벨링 방법임)
def _get_area(box):
  return box[2] * box[3]

# 교집합을 구하는 함수
def _get_intersection_area(box1,box2):
  return abs(max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2])) *abs(
      max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3])
  )

# 합집합의 영역을 구하는 함수
def _get_union_area(box1, box2, inter_area=None):
  area_a = _get_area(box1)
  area_b = _get_area(box2)
  if inter_area is None:
    inter_area = _get_intersection_area(box1,box2)
  return float(area_a + area_b - inter_area)

# IOU를 계산하는 함수
def iou(box1, box2):
  # if boxes do no intersect
  if _is_box_intersect(box1, box2) is False:
    return 0
  ineter_area = _get_intersection_area(box1,box2)
  union = _get_union_area(box1,box2, inter_area=ineter_area)

  # intersection over union
  iou = ineter_area / union
  if iou < 0:
    iou = 0
  assert iou >=0, f'Measuer is wronf! L IoU Value is [{iou}]'
  return iou

box1 = (0.3,0.3,0.1,0.1)
box2 = (0.31,0.28,0.14,0.13)
print(iou(box1,box2))

import cv2
import numpy as np
import matplotlib.pyplot as plt

def nms(boxes,iou_thres=0.4):
  elems = np.array(boxes)
  print('\nBefore Arrange')
  print(elems)

  sorted_index = np.argsort(elems[:,-1][::-1])
  sorted_boxes = elems[sorted_index]

  print('\nAfter Arrange')
  print(sorted_boxes)

  answer = [True for _ in range(sorted_boxes.shape[0])]
  print('\nBefore NMS Answer :', answer)

  for i in range(sorted_boxes.shape[0]):
    if not answer[i]:
      continue
    for j in range(i+1, sorted_boxes.shape[0]):
      iou_val = iou(sorted_boxes[i], sorted_boxes[j])
      print(f"{i} vs {j} =  iou {round(iou_val, 3)}")
      if iou_val >= iou_thres:
        answer[j]= False
        print(f'index {j} is False')
  print('\nAfter NMS Answer :', answer)
  return answer, sorted_boxes,sorted_index

colorset = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

boxes = [
    [0.3, 0.3, 0.1, 0.1, 0.9],
    [0.31, 0.28, 0.14, 0.13, 0.5],
    [0.28, 0.28, 0.09, 0.11, 0.3],
    [0.75, 0.65, 0.2, 0.2, 0.99],
    [0.7, 0.63, 0.22, 0.18, 0.35],
    [0.75, 0.62, 0.22, 0.22, 0.77],
]

from re import S
width, height = 600,600
canvas = np.zeros((width, height,3), dtype=np.uint8)
canvas_copy = canvas.copy()


for index, box in enumerate(boxes):
  pt1 = (int(width * box[0] - width * box[2] / 2), int(height * box[1] - height * box[3] / 2))
  pt2 = (int(width * box[0] + width * box[2] / 2), int(height * box[1] + height * box[3] / 2))

  cv2.rectangle(canvas, pt1, pt2, colorset[index], 2)

answer, sorted_boxes, sorted_index = nms(boxes, iou_thres=0.4)
for index, (sbox,sidx) in enumerate(zip(sorted_boxes, sorted_index)):
  if answer[index]:
      cx, cy, bw, bh = sbox[0], sbox[1], sbox[2], sbox[3]
      pt1 = (int(width * (cx - bw / 2)), int(height * (cy - bh / 2)))
      pt2 = (int(width * (cx + bw / 2)), int(height * (cy + bh / 2)))

      cv2.rectangle(canvas_copy, pt1, pt2, colorset[sidx], 2)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axs[0].set_title('before NMS')
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(canvas_copy, cv2.COLOR_BGR2RGB))
axs[1].set_title('after NMS')
axs[1].axis('off')
plt.show()
# 가장 강한 녀석만 살아 남는다 컨피던스 스코어가 높은

!pip install ultralytics

from google.colab import files
uploaded = files.upload()

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 모델 로드
model = YOLO('yolo11n.pt')
# model = YOLO('yolov8n-seg.pt')
# 이미지 예측
image_path = 'bts.jpg'
results = model.predict(image_path)

# 시각화된 결과 이미지 얻기
res_img = results[0].plot()  # 반드시 () 호출해야 함!

# 결과 출력
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('YOLO11 n Detection Result')
plt.show()

