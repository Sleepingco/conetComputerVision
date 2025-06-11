import cv2
import numpy as np
import os
from PIL import Image
import dlib

labels = ["Cha","Ma"]

# hog_face_detector =  dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainer_CNN.yml')

image_list = []

test_images = os.path.join(os.getcwd(), 'test_list')
for root, dirs, files in os.walk(test_images):
    for file in files:
        if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
            image_path = os.path.join(test_images, file)
            print(image_path)
            image_list.append(cv2.imread(image_path))
for img in image_list:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # faces = hog_face_detector(gray,1)
    faces = cnn_face_detector(gray,1)
    for face_detection in faces:
        x,y,w,h = face_detection.rect.left(),face_detection.rect.top(),face_detection.rect.right(),face_detection.rect.bottom()
        roi_gray = gray[y:y+h,x:x+w]

        id_, conf =  recognizer.predict(roi_gray)
        print(labels[id_],conf)

        if conf>=50:
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            name = labels[id_]
            cv2.putText(img,name,(x,y),font,1,(0,0,255),2)
            cv2.rectangle(img,(x,y,),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Preview',img)
    if cv2.waitKey(0) >=0:
        continue
cv2.destroyAllWindows()

# 실행 결과 확인! CNN은 엄청 오래걸림