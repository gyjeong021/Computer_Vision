import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.4,fy=0.4)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.putText(gray,'soccer',(10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
cv.imshow('Original',gray)

# 가우시안 블러 적용 (입력 영상, 적용할 필터 크기, 시그마 값(얼마나 영상 부드럽게 할지, 기본값0))
# 필터 크기 클수록 블러링 정도 커짐
# 가우시안 블러 opencv에서 함수 제공
smooth=np.hstack((cv.GaussianBlur(gray,(5,5),0.0),cv.GaussianBlur(gray,(9,9),0.0),cv.GaussianBlur(gray,(15,15),0.0)))
cv.imshow('Smooth',smooth)

# 엠보싱 3*3 필터 정의
femboss=np.array([[-1.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 1.0]])

gray16=np.int16(gray) # gray는 1바이트(8bits) => 16bits (-255~255)
emboss=np.uint8(np.clip(cv.filter2D(gray16,-1,femboss)+128,0,255)) # 0보다 작으면 0으로, 255보다 크면 255로
emboss_bad=np.uint8(cv.filter2D(gray16,-1,femboss)+128) # (-127~383)
emboss_worse=cv.filter2D(gray,-1,femboss)

cv.imshow('Emboss',emboss)
cv.imshow('Emboss_bad',emboss_bad)
cv.imshow('Emboss_worse',emboss_worse)

cv.waitKey()
cv.destroyAllWindows()