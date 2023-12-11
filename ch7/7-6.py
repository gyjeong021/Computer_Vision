import numpy as np
import tensorflow as tf
import keras.datasets as ds

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 1) 데이터 수집
(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
x_train=x_train.reshape(50000,3072)
x_test=x_test.reshape(10000,3072)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

# 2) 모델 선택 : MLP(다층 퍼셉트론)
dmlp=Sequential()
dmlp.add(Dense(units=1024,activation='relu',input_shape=(3072,)))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=10,activation='softmax')) # 마지막 add가 출력층

# 3) 학습
dmlp.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
hist=dmlp.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=2)

# 4) 예측
print('정확률=', dmlp.evaluate(x_test,y_test,verbose=0)[1]*100)

# dmlp.save('dmlp_trained.h5')

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
plt.show()

import cv2 as cv

class_names=['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']

img = cv.imread('dog0000.jpg')
print(img.shape)

img = cv.resize(img, dsize=(32,32))
img = img.reshape(1,3072) # 1차원으로
img_test = img.astype(np.float32)/255.0 # 0~1 정규화

y_predict = dmlp.predict(img_test)
print(y_predict[0])
class_idx = np.argmax(y_predict[0])
print(class_idx,class_names[class_idx])