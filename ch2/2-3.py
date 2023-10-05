import cv2 as cv
import sys

img=cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)	# BGR 컬러 영상을 명암(흑백) 영상으로 변환
gray_small=cv.resize(gray,dsize=(0,0),fx=0.5,fy=0.5) # 반으로 축소
# dsize : 원하는 크기가 있다면 (크기, 크기), 없다면 (0,0) 하고 비율 표시

cv.imwrite('soccer_gray.jpg',gray)	# 영상을 파일에 저장 
cv.imwrite('soccer_gray_small.jpg',gray_small)  
    
cv.imshow('Color image',img)
cv.imshow('Gray image',gray)
cv.imshow('Gray image small',gray_small)

cv.waitKey()
cv.destroyAllWindows()