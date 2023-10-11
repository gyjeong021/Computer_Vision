import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')	 # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 그레이 변환

canny=cv.Canny(gray,100,200) # 엣지 이미지

t,bin_gray=cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # 이진 영상

# 경계선에 대한 정보, 대칭 관계에 대한 정보
# contour,hierarchy=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
contour,hierarchy=cv.findContours(bin_gray,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

lcontour=[]   
for i in range(len(contour)):
    if contour[i].shape[0]>100:	# 포함되어 있는 엣지 픽셀의 개수가 100보다 크면
        lcontour.append(contour[i])
    
# cv.drawContours(img,lcontour,-1,(0,255,0),3) # 길이 100 이상 contour만
cv.drawContours(img,contour,-1,(0,255,0),3) # 모든 contour(길이 100이하 포함)
             
cv.imshow('Original with contours',img)    
# cv.imshow('Canny',canny)
cv.imshow('Otsu binarization',bin_gray)

cv.waitKey()
cv.destroyAllWindows()