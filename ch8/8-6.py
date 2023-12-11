import cv2 as cv 
import numpy as np
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions

model=ResNet50(weights='imagenet') # ResNet50으로 되어있는 imagenet이라는 데이터셋을 불러옴

# img=cv.imread('rabbit.jpg')
img=cv.imread('dog1.jpg')
x=np.reshape(cv.resize(img,(224,224)),(1,224,224,3)) # 크기 : (224,224)
x=preprocess_input(x) # 0~1 정규화하는 과정 함수로 처리

preds=model.predict(x)
top5=decode_predictions(preds,top=5)[0] # top-5 Accuracy까지 보겠음
print('예측 결과:',top5)

for i in range(5):
    cv.putText(img,top5[i][1]+':'+str(top5[i][2]),(10,20+i*20),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

cv.imshow('Recognition result',img)

cv.waitKey()
cv.destroyAllWindows()