import numpy as np
import tensorflow as tf
import keras.datasets as ds # keras 별도 라이브러리

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.optimizers import Adam

# 1) 데이터 수집
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000,28,28,1) # MLP는 (60000,784), CNN은 1차원으로 변환 과정 없음(가장 큰 차이)
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype(np.float32)/255.0 # 0~1 정규화
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # onehot 인코딩
y_test=tf.keras.utils.to_categorical(y_test,10)

# 2) 모델 선택 : LeNet5
cnn=Sequential()
cnn.add(Conv2D(6,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))  # (5,5) 필터 이용
# padding : same - 입력영상에 버퍼를 집어넣어 입력영상과 출력영상의 크기 동일
# input (28*28), output (28*28)
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
# output (14*14)
cnn.add(Conv2D(16,(5,5),padding='valid',activation='relu')) # (5,5) 필터 이용
# padding : valid - 입력영상을 그대로 이용해서 필터가 적용된 크기, 필터 크기에 비례해 줄어든 출력영상의 크기
# output (10*10), (2*2) 필터에 의해 크기 줄어듬
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
cnn.add(Conv2D(120,(5,5),padding='valid',activation='relu')) # (5,5) 필터 이용
cnn.add(Flatten()) # 앞에서의 텐서 형태의 구조를 1차원 구조로 만들어줌
cnn.add(Dense(units=84,activation='relu')) # Dense 층에 넣으려면 1차원 배열로 만들어줘야함
cnn.add(Dense(units=10,activation='softmax'))

cnn.summary()

cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy']) 
cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)
                                    
res=cnn.evaluate(x_test,y_test,verbose=0) 
print('정확률=',res[1]*100)


# cnn 필터의 모습 보이기
import matplotlib.pyplot as plt

# 세번째 layer의 가중치 불러오기
filters, bias = cnn.layers[2].get_weights()
print(filters.shape)

# 0~1 정규화
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

f = filters[:,:,:,0]
fig = plt.figure(figsize=(2,2))
plt.imshow(f[:,:,0], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

print(f[:,:,0])

# 여러 개의 필터 보이기
n_filters, ix = 4, 1
fig = plt.figure(figsize=(8,8))
for i in range(n_filters):
    f = filters[:,:,:,0]
    for j in range(4):
        plt.subplot(n_filters,4,ix)
        plt.imshow(f[:,:,j],cmap='gray')
        ix+=1
plt.show()