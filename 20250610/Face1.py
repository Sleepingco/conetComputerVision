import cv2
import numpy as np
import os
from PIL import Image
import dlib
# Dlib이란?
# OpenCV와 유사한 라이브러리
# 주로 얼굴 탐지(detection)와 정렬(alignment) 모듈을 사용
# dlib는 즉시 사용가능한 강력한 얼굴인식(recognition) 모듈도 제공한다. 비록 C++로 작성되었지만 파이썬 인터페이스도 가지고 있음 (즉, 속도가 매우 빠름)
import time

image = cv2.imread('c:/Users/main/Downloads/AI_Jin/source/ch2/people.jpg')
image_resized = cv2.resize(image, (755,500))

cascade_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 캐스케이드 검출기의 scaleFactor 를 사용한 튜닝
# sacleFactor 는 얼굴을 검출할 사이즈를 설정 할  수 있음 (이미지에 따라서 얼굴이 크게 나오거나 적게 나올수 있기 때문에 이런 설정을 사용)
# 기본값은 1.1 인데 1.01 으로 바꿔서 실행해보자!
# minNeighbors 파라미터를 설정
# 캐스케이드 검출기로 검출을 하면 얼굴 주변에 여러 개의 후보 경계 박스(candidate bounding boxes)를 생성
# 여러 후보 경계 박스 가운데 가장 얼굴을 잘 둘러싸는 경계 박스를 최종적으로 선택합
# minNeighbors 파라미터는 최종 경계 박스를 선택하기 위해 얼굴 주변에 존재해야 하는 최소 후보 경계 박스 개수입니다. 만약에 minNeighbors=5라면 한 얼굴에 최소한 5개의 후보 경계 박스가 있어야 해당 얼굴을 검출을 의미

face_detections = cascade_face_detector.detectMultiScale(image_resized,scaleFactor=1.01,minNeighbors=2)
print(face_detections)

for (x,y,w,h) in face_detections:
    cv2.rectangle(image_resized, (x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('test', image_resized)

# 기본 하이퍼 파라미터 실제로는 5명이 있는데, 경계 박스는 4개를 그림 – 22년전 초기 알고리즘이라서 정확도 문제가 많음

cv2.waitKey(0)
# cv2.destroyAllWindows()

# hog 이용
image = cv2.imread('c:/Users/main/Downloads/AI_Jin/source/ch2/people.jpg')
image = cv2.imread('c:/Users/main/Downloads/AI_Jin/source/ch2/bts.jpg')
# 결과 고찰 : 검출은 되었는데 선글라스를 낀 뒷사람이 인식이 안됨?
# 선글라스에 대한 HOG 가 일반 사용자의 눈 모습과 달라서 생기는 오류!
# CNN 으로 얼굴 검출이 더 빠르지 않을까? 
start = time.time()
image_resized = cv2.resize(image, (755,500))
hog_face_detector = dlib.get_frontal_face_detector()
face_detections =  hog_face_detector(image_resized,1)

print(face_detections)
end = time.time()
for face_detection in face_detections:
    left, top, right, bottom = face_detection.left(), face_detection.top(), face_detection.right(), face_detection.bottom() # 하르 캐스캐이드 는 리스트를 반환해서 일일히 써줘야 했지만 hog는 함수를 쓰면 반환하면 그냥 받는다
    cv2.rectangle(image_resized,(left,top),(right,bottom),(0,255,0),2)
print(f'HOG{end-start}초')
cv2.imshow('test', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

start = time.time()
image_resized_CNN = image_resized.copy()

cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
face_detections_CNN = cnn_face_detector(image_resized_CNN, 1)

print(face_detections_CNN) # <_dlib_pybind11.mmod_rectangles object at 0x0000017CFAB4A230> 빌트인 함수
end = time.time()
for idx,face_detection_CNN in enumerate(face_detections_CNN):
    left, top, right, bottom,confidence = face_detection_CNN.rect.left(), face_detection_CNN.rect.top(), face_detection_CNN.rect.right(),face_detection_CNN.rect.bottom(),face_detection_CNN.confidence
    print(f'confidence{idx+1}:{confidence}') # print confidence of the detection
    cv2.rectangle(image_resized_CNN,(left,top),(right,bottom),(0,255,0),2)
print(f'CNN{end-start}초')
cv2.imshow("CNN",image_resized_CNN)
cv2.waitKey(0)
cv2.destroyAllWindows()

# hog 0.3s cnn 15.7s