import cv2
import numpy as np
import os
from PIL import Image
import dlib # hog인식을 위한 dlib 라이브러리 추가

# HOG 인식으로 변경하기 어제는 하르 케스케이드
hog_face_detector = dlib.get_frontal_face_detector()
recognizer = cv2.face.LBPHFaceRecognizer_create() # LBPH를 사용할 새 변수 생성

Face_ID = -1
pev_person_name = ""
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "Face_Images")
print(Face_Images)
for root, dirs, files in os.walk(Face_Images):
    for file in files :
        if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
            path = os.path.join(root,file)
            person_name = os.path.basename(root)
            print(path, person_name)

            if pev_person_name != person_name: # 이름이 바뀌었는지 확인
                Face_ID=Face_ID+1
                pev_person_name = person_name

            img = cv2.imread(path) #이미지 파일 가져오기
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = hog_face_detector(gray_image,1)

            print(Face_ID,faces)

            for face_detection in faces:
                x,y,w,h, = face_detection.left(), face_detection.top(),face_detection.right(),face_detection.bottom()
                roi = gray_image[y:y+h,x:x+w] # 얼굴 부분만 가져오기
                x_train.append(roi)
                y_ID.append(Face_ID)

recognizer.train(x_train,np.array(y_ID))
recognizer.save('face-trainer.yml')