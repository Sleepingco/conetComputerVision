import cv2
import dlib
import math
import time

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print('could not open webcam')
    exit()

# 좀더 최적화를 한다면?
# 디스크립터는 웹캠 열기전에 만들고 실시간 추론(inference)만 진행하기!
# 하지만 CNN은 너무 느림 ㅠㅠ, CNN 정도의 망으로 실시간으로 하고 싶으면 C# 혹은 C++ 프로그래밍을 사용해야함!

cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

while webcam.isOpened():
    status, frame = webcam.read()
    image_resized = cv2.resize(frame,(755,500))
    # # HOG 검출기로 얼굴을 인식해보자
    # start = time.time()
    # hog_face_dectector = dlib.get_frontal_face_detector()
    # face_detections = hog_face_dectector(image_resized,1)
    # end = time.time()
    # for face_detection in face_detections:
    #     left, top, right, bottom = face_detection.left(),face_detection.top(),face_detection.right(),face_detection.bottom()
    #     cv2.rectangle(image_resized,(left, top),(right,bottom),(0,255,0),2)
    
    # org=(50,100)
    # text = f'{end - start:.5f} HOSs'
    # font=nt=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    # cv2.putText(image_resized,text,org,font,1,(255,0,0),2)

    # CNN 검출기로 얼굴을 인식해보자
    start = time.time()
    face_detections_CNN =  cnn_face_detector(image_resized,1)
    end = time.time()
    for idx, face_detection_CNN in enumerate(face_detections_CNN):
        left, top, right, bottom, confidence = (
            face_detection_CNN.rect.left(),
            face_detection_CNN.rect.top(),
            face_detection_CNN.rect.right(),
            face_detection_CNN.rect.bottom(),
            face_detection_CNN.confidence
        )

        print(f'confidence{idx+1}: {confidence}') # print confidence of the detection
        cv2.rectangle(image_resized,(left,top),(right,bottom),(0,255,0),2)
    org=(50,100)
    text = f"{end-start:.5f} CNNs"
    font=cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image_resized,text,org,font,1,(255,0,0),2)
    if status:
        cv2.imshow("test",image_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
webcam.release()
cv2.destroyAllWindows()
# CNN은 파이썬으로 만들어짐 그래서 시간이 오래걸림