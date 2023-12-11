import tensorflow as tf
import keras.datasets as ds
import matplotlib.pyplot as plt

# x_train : 입력 영상 학습 데이터, y_train : 입력 영상의 라벨 값(0~9)
# x_test : 테스트 데이터, y_test : 테스트 라벨 값
# 훈련과 학습은 중복되면 안됨
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data() # mnist : 0~9까지의 필기체
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
plt.figure(figsize=(24,3))
plt.suptitle('MNIST',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title(str(y_train[i]),fontsize=30)

print(x_train[0], y_train[0]) # x_train 28*28 크기 배열로 이루어짐

(x_train,y_train),(x_test,y_test)=ds.fashion_mnist.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
fashion_class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(24,3))
plt.suptitle('MNIST',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title(str(y_train[i]) + fashion_class_names[y_train[i]], fontsize=20)

(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
class_names=['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(24,3))
plt.suptitle('CIFAR-10',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[y_train[i,0]],fontsize=30)