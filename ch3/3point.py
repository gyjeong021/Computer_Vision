import cv2 as cv
import sys
import numpy as np

img=cv.imread('soccer.jpg')

# cvtColor를 사용하지 않고 imread로 읽을 떄 바로 영상 변환할 수도 있음
# gray=cv.imread('soccer.jpg', cv.IMREAD_GRAYSCALE) # BGR 컬러 영상을 명암 영상으로 변환하여 저장


if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)	# BGR 컬러 영상을 명암 영상으로 변환
gray_small=cv.resize(gray,dsize=(0,0),fx=0.25,fy=0.25) # 이미지 1/4로 축소

cv.imshow('soccer-gray',gray_small)
print(gray_small.shape)

# 사칙연산
img_plus = cv.add(gray_small, 50)         # y = x + 50
img_minus = cv.subtract(gray_small, 50)   # y = x - 50
img_multi = cv.multiply(gray_small, 2)    # y = 2 * x
img_div = cv.divide(gray_small, 2)        # y = x / 2
pp=np.hstack((img_plus, img_minus, img_multi, img_div)) #hstack은 높이(세로)가 같아야 함
cv.imshow('point processing',pp)

# 사칙연산 연산자
# 연산자를 사용하면 값이 255가 넘어갔을 떄, 클램핑 처리를 하지 못해 자연스럽게 영상 처리를 못함
# 연산자 대신 opencv 사용하기
img_plus2 = gray_small+50       # y = x + 50
img_minus2 = gray_small-50      # y = x - 50
img_multi2 = gray_small*2       # y = 2 * x
img_div2 = gray_small/2         # y = x / 2
cv.imshow('test',img_plus2)

print(gray_small[100,100], img_plus[100,100], img_plus2[100,100]) # 72
print(gray_small[180,80], img_plus[180,80], img_plus2[180,80]) # 226

#cv.imshow('test',img_minus2)
#print(gray_small[100,100], img_minus[100,100], img_minus2[100,100]) # 72
#print(gray_small[120,180], img_minus[120,180], img_minus2[120,180]) # 27

img512=cv.resize(img,dsize=(512,512)) # (512,512) 크기로 조정
opencv_img=cv.imread('opencv_logo512.png')
img_plus3 = cv.add(img512, opencv_img)
cv.imshow('two images - add',img_plus3)

# img512를 0.5만큼 줄이고 opencv_img를 0.5만큼 줄이기, gamma는 별다른 조건 없으면 0
# alpha와 beta는 두 값을 더해서 1이 되는 수
img_plus4 = cv.addWeighted(img512, 0.5, opencv_img, 0.5, 0)
cv.imshow('two images - addWeighted',img_plus4)

img_rev = cv.subtract(255,gray_small)     # y = 255 - x
cv.imshow('reverse image', img_rev)

# 임계값 50
ret, img_binary50 = cv.threshold(gray_small, 50, 255, cv.THRESH_BINARY)
cv.imshow('threshold',img_binary50)

ret, img_binary100 = cv.threshold(gray_small, 100, 255, cv.THRESH_BINARY)
ret, img_binary150 = cv.threshold(gray_small, 150, 255, cv.THRESH_BINARY)
ret, img_binary200 = cv.threshold(gray_small, 200, 255, cv.THRESH_BINARY)
img_binary=np.hstack((img_binary50, img_binary100, img_binary150, img_binary200))
#cv.imshow('threshold',img_binary)

ret, img_binaryB = cv.threshold(gray_small, 100, 255, cv.THRESH_BINARY)
ret, img_binaryBINV = cv.threshold(gray_small, 100, 255, cv.THRESH_BINARY_INV)
ret, img_binaryT = cv.threshold(gray_small, 100, 255, cv.THRESH_TRUNC)
ret, img_binaryT0 = cv.threshold(gray_small, 100, 255, cv.THRESH_TOZERO)
ret, img_binaryT0INV = cv.threshold(gray_small, 100, 255, cv.THRESH_TOZERO_INV)
img_binary2=np.hstack((img_binaryB, img_binaryBINV, img_binaryT, img_binaryT0, img_binaryT0INV))
cv.imshow('threshold',img_binary2)

cv.waitKey()
cv.destroyAllWindows()