from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D, Flatten,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

train_generator = ImageDataGenerator(rotation_range=10,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     rescale=1/255)

train_dataset = train_generator.flow_from_directory(directory='archive/train',
                                                    target_size=(48,48),
                                                    class_mode='categorical',
                                                    batch_size=16,
                                                    shuffle=True,
                                                    seed=10)

test_generator = ImageDataGenerator(rescale=1/255)
test_dataset = test_generator.flow_from_directory(directory='archive/test',
                                                  target_size=(48,48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)

# FER 데이터셋은 7가지의 감정 분류를 가짐, 디텍터는 32개 이미지는 48X48 셋을 가지고 있음
num_classes = 7
num_detectors = 32
width, height = 48, 48

# 이를 반영하는 망을 만들어 보도록 하자
network = Sequential()
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same', input_shape=(width, height, 3))) # 특징 추출
network.add(BatchNormalization()) # 정규화
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2))) # 작게 만듬
network.add(Dropout(0.2)) # 일부 뉴런 드롭아웃 과적합 방지를 위해 여기 까지 한 층
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Flatten())
network.add(Dense(2*2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))
network.add(Dense(2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))
network.add(Dense(num_classes, activation='softmax'))
network.summary()

network.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_val_acc = EarlyStopping(monitor='val_accuracy',patience=5)
filename = 'emotions_best.h5'
checkpoint = ModelCheckpoint(filename,
                             verbose=1,
                             save_best_only=True
                             )
epochs = 70
history=network.fit(train_dataset,epochs=epochs,validation_data=test_dataset,callbacks=[checkpoint,model_val_acc]).history
print('학습 종료')
network.save(filename)

score = network.evaluate(test_dataset)
print('Test loss', score[0])
print('Test accuracy', score[1]*100)

import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history['accuracy'])
plt.plot(history('val_accuracy'))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel("epochs")
plt.legend(['train','validation'],loc='upper left')
