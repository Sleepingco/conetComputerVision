import random
import numpy as np

X=[5,3,5,5,2,4,1,6,3,1]  # 데이터셋

eyes=[1,2,3,4,5,6]  # 주사위 눈
p=np.zeros([6])

def learn_generator(X,p): # 생성 모델 학습
    for i in range(len(X)): 
        p[X[i]-1]+=1
    p=p/len(X)
    
def generate(): # 생성
    return(random.choices(eyes,p))

learn_generator(X,p)
print(generate(),generate(),generate(),generate()) # 네 번 생성
