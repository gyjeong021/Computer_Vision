import cv2 as cv

img=cv.imread('rose.png')
patch=img[250:350,170:270,:]

img=cv.rectangle(img,(170,250),(270,350),(255,0,0),3)
# 크기 변환 : opencv에서 resize 함수 제공
# fx, fy : 얼만큼 확대, 축소할지
# interpolation : 어떤 방식으로 색상 결정할지
patch1=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)
patch2=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR)
patch3=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)

cv.imshow('Original',img)
cv.imshow('Resize nearest',patch1) 
cv.imshow('Resize bilinear',patch2) 
cv.imshow('Resize bicubic',patch3) 

cv.waitKey()
cv.destroyAllWindows()