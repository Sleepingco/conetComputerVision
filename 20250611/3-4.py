from deepface import DeepFace
from deepface.basemodels import VGGFace,OpenFace,Facenet,FbDeepFace

import matplotlib.pyplot as plt
import numpy as np
import cv2

# 다음과 같이 Alignment 테스트를 해보기!
# 원래 휘어져 있던 이미지가 제대로 되는 거를 확인할 수 있음!
# DeepFace 설치 및 코딩 실행 (임포트 할때 대소문자 주의!) 총 4개의 모델을 가지고 있음
# VGG-Face는 (224, 224, 3) 모양으로 입력을 받고 2622차원 벡터로 출력
# Google FaceNet은 (160, 160, 3)으로 입력하여 128차원 배열로 출력
# detectFace함수에 입력 모양을 전달해야  하기 때문에 input_size변수 선언
# 딥페이스(deepface)는 함수로 얼굴탐지와 얼굴 정렬 모두를 제공
# 딥페이스는 opencv의 haar cascade, SSD, dlib HoG와 MTCNN을 포함
# 얼굴을 정렬하기 위해 몇가지 수학과 삼각법을 할 수 있음
# 우리는 이미지 경로만 전달 하면 됨 만약 코드의 'detector_backend' 인자를 사용하지 않으면 기본 설정인 opencv의 haar cascade를 사용

# model = VGGFace.loadModel()
model = Facenet.loadModel()
# model = OpenFace.loadModel()
# model = FbDeepFace.loadModel()



input_size = model.layers[0].input_shape[1:3] # VGGFace
input_size = model.input_shape[1:3] # Facenet

backends = ['opencv','ssd','dlib','mtcnn']

img1 = DeepFace.detectFace('aj1.jpg',detector_backend = backends[3])
img2 = DeepFace.detectFace('aj2.png',detector_backend = backends[3])
img1 = cv2.resize(img1,input_size)# Facenet
img2 = cv2.resize(img2,input_size)# Facenet
img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)
img1_representation = model.predict(img1)[0,:]
img2_representation = model.predict(img2)[0,:]

distance_vector = np.square(img1_representation - img2_representation)
distance = np.sqrt(distance_vector.sum())
print('Euvlideean distance : ',distance) # 두 점 사이의 거리를 계산하는 기법

img1_graph = []
img2_graph = []
distance_graph = []
for i in range(0,200):
    img1_graph.append(img1_representation)
    img2_graph.append(img2_representation)
    distance_graph.append(distance_vector)

img1_graph = np.array(img1_graph)
img2_graph = np.array(img2_graph)
distance_graph =np.array(distance_graph)

fig = plt.figure()
ax1 = fig.add_subplot(3,2,1)
# plt.imshow(img1[0][:,:,::-1]) BGR일 경우, RGB로 바꿀때 사용
plt.imshow(img1[0])
plt.axis('off')
ax2=fig.add_subplot(3,2,2)
im = plt.imshow(img1_graph, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
ax3 = fig.add_subplot(3,2,3)
plt.imshow(img2[0])
plt.axis('off')
ax4 =  fig.add_subplot(3,2,4)
im = plt.imshow(img2_graph,interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
ax5=fig.add_subplot(3,2,5)
plt.text(0.35,0,'Distande: %s' % (distance))
plt.axis('off')
ax6 =  fig.add_subplot(3,2,6)
im = plt.imshow(distance_graph,interpolation='nearest',cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 딥러닝을 사용한 접근을 한다면..? CNN 으로 얼굴을 추적후 LBPH 을 적용한 인식?
# 이 방법을 사용하는 것보다는.. 딥러닝을 적용한 Face 오픈 소스를 공부하고 적용하는 것이 훨씬 더 좋음 (정확도 측면에서!)  
# 인식 알고리즘의 설계?
# 처음에 predict 를 할 사람 얼굴에 대한 검사를 수행후 해당 결과 정보들을 바탕으로 저장을 수행 (딕셔너리등등)
# 이후 나오는 이미지들에 따라서 임계값보다 낮은 경우는 동일 인물 아닌 경우는 다른 인물로 처리
# 예제로 드리는 다른 유명인들 (도널드 트럼프, 버락 오바마) 등을 해보고 괜찮다 싶으면 한국인 이미지도 해보자 
# DeepFace 학습에 동양인들도 들어가 있어서 학습이 어느정도 수행될 수 있음
