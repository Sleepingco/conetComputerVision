import cv2
import numpy as np
import os
from PIL import Image

labels = ['Carina','Cha','Ma','Win']


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainer2.yml')
image_list = []
test_images = os.path.join(os.getcwd(), 'test_list')

for root, dirs, files in os.walk(test_images):
    for file in files:
        if file.endswith("jpeg") or file.endswith('jpg') or file.endswith('png'):
            image_path = os.path.join(test_images, file)
            print(image_path)
            image_list.append(cv2.imread(image_path))
# 이제 test_list 폴더에 있는 이미지들을 검색 하면서 먼저 HarrCaseCade 방법으로 얼굴 검출
# 이미지를 가져와서 LBPH 인식기로 인식 및 해당 값을 출력
# 계속 이어짐 이후에는 cv 로 이미지를 보여주고 아무키나 누르면 다음 이미지로 넘어감 그후 다 끝나면 종료 하도록 설정

for img in image_list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        print(labels[id_],conf)
    
        if conf>=50:
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            name = labels[id_]
            cv2.putText(img,name,(x,y),font,1,(0,0,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Preview',img)
    if cv2.waitKey(0) >=0:
        continue
cv2.destroyAllWindows()