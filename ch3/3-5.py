import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg') 
img=cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25) # 1/4로 축소

def gamma(f,gamma=1.0):
    f1=f/255.0			# L=256이라고 가정, 0~1 사이의 값으로
    return np.uint8(255*(f1**gamma)) # 0~255 사이의 값으로, 영상은 반드시 정수 -> 게산된 값을 uint8을 사용하여 1byte 정수로 바꿔줌

gc=np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),gamma(img,2.0),gamma(img,3.0)))
cv.imshow('gamma',gc)

cv.waitKey()
cv.destroyAllWindows()