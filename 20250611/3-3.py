import cv2
import math
import time

labels = ["Cha","Ma"]
webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainer.yml")

if not webcam.isOpened():
    print("could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 하르 캐스캐이드 검출기로 얼굴 인식
    start =time.time()
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    end = time.time()

    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        id_,conf = recognizer.predict(roi_gray)
        print(labels[id_],conf)

        if conf>=50:
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            name = labels[id_]
            cv2.putText(frame, name,(x,y),font,1,(0,0,255),2)
            org = (50,100)
            text = f'{end-start:.5f} second'
            cv2.putText(frame,text,org,font,1,(255,0,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    if status:
        cv2.imshow('test',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()