# 다룸 가능 가우시안을 이용한 생성 모델
import numpy as np

X=np.array([[168,70],[172,68],[175,78],[163,68],[180,80],[159,76],[158,52],[173,69],[180,75],[155,50],[187,80],[170,66]])

m=np.mean(X,axis=0)
cv=np.cov(X,rowvar=False)

gen=np.random.multivariate_normal(m,cv,5)

print(gen) # 데이터 셋과 유사한 샘플 생성됨