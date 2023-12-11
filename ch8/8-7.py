from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Rescaling
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121
from keras.utils import image_dataset_from_directory
import pathlib

# 1) 데이터 준비
data_path=pathlib.Path('datasets/stanford_dogs/images/images')

# validation_split = 0.2는 테스트(검증)에 0.2, 나머지 0.8을 학습에 쓰겠다는 의미
# batch_size = 16은 한개의 그룹 당 16개씩 나눠서 학습하겠다는 의미
train_ds=image_dataset_from_directory(data_path,validation_split=0.2,subset='training',seed=123,image_size=(224,224),batch_size=16)
test_ds=image_dataset_from_directory(data_path,validation_split=0.2,subset='validation',seed=123,image_size=(224,224),batch_size=16)

class_names = train_ds.class_names
print(class_names)

# 2) 모델 선택
base_model=DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3))

cnn=Sequential()
cnn.add(Rescaling(1.0/255.0)) # 0~1 정규화
cnn.add(base_model) # 특징 추출 : 사전학습 모델
cnn.add(Flatten())
cnn.add(Dense(1024,activation='relu')) # 분류 : 직접 학습시킨 모델
cnn.add(Dropout(0.75))
cnn.add(Dense(units=120,activation='softmax')) # output 층

base_model.summary()

cnn.build(input_shape=(None,224,224,3))
cnn.summary()

for images, labels in train_ds.take(1): # take(1) : 데이터 한 세트 가져오기
    print('images shape:', images.numpy().shape)
    print('Labels shape:', labels.numpy().shape)
    print('Label:', labels.numpy())
    break

# 3) 학습
cnn.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.000001),metrics=['accuracy'])
# sparse_categorical_crossentropy : one-hot 인코딩이 아닌 정수 형태로 표시되어 있는 경우
hist=cnn.fit(train_ds,epochs=200,validation_data=test_ds,verbose=2)

# 4) 예측
print('정확률=',cnn.evaluate(test_ds,verbose=0)[1]*100)

cnn.save('cnn_for_stanford_dogs.h5')	# 미세 조정된 모델을 파일에 저장

import pickle
f=open('dog_species_names.txt','wb')
pickle.dump(train_ds.class_names,f)
f.close()

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()