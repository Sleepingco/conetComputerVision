import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import dlib

test_generator = ImageDataGenerator(rescale=1/255)
test_dataset = test_generator.flow_from_directory(directory='archive/test',
                                                  target_size=(48,48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)

network = load_model('emotion_best.h5')

image_list = []
test_images = os.path.join(os.getcwd(),'test_list')

face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

for root, dirs, files in os.walk(test_images):
    for file in files:
        if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
            image_path = os.path.join(test_images,file)
            print(image_path)
            image_list.append(cv2.imread(image_path))
            # 다음 소스코드는 이미지리스트에 읽어 들인 이미지 한장에서 사람들의 얼굴을 찾고 (dlib의 얼굴 검출기 사용)
            # 찾은 얼굴 영역만 (roi) 감정 인식 모듈에 prediction을 시킨후 7가지 감정 정보들 중에서 가장 높은 값에 해당하는 대표 감정을 name 변수에 담아서 이를 텍스트 출력해주는 함수
            # 사진이 여러장인 경우 아무키나 누르면 다음 사진으로 이동하게됨 (시간이 걸림)

for img in image_list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img,1)
    for face_detection in faces:
        left,top = face_detection.rect.left(),face_detection.rect.top()
        right,bottom = face_detection.rect.right(),face_detection.rect.bottom()
        roi = img[top:bottom,left:right]
        roi =  cv2.resize(roi,(48,48))
        roi = roi / 255
        roi = np.expand_dims(roi,axis=0)
        pred_probability = network.predict(roi)
        print(pred_probability)
        print(np.argmax(pred_probability))
        i = 0
        name =  ''
        for index in test_dataset.class_indices:
            print(index)
            if i == np.argmax(pred_probability):
                name = index
            i +=1
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(img,name,(left,top),font,1,(0,0,255),2)
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
    cv2.imshow('preview', img)
    if cv2.waitKey(0) >= 0:
        continue