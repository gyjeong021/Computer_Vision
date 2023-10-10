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

# 평균값 3*3 필터 정의
faverage=np.array([[1.0/9.0, 1.0/9.0, 1.0/9.0],
                  [ 1.0/9.0, 1.0/9.0, 1.0/9.0],
                  [ 1.0/9.0, 1.0/9.0, 1.0/9.0]])

# 샤프닝 3*3 필터 정의 (경계선만 표시)
fsharping=np.array([[0.0, -1.0, 0.0],
                  [ -1.0, 4.0, -1.0],
                  [ 0.0, -1.0, 0.0]])

# 샤프닝 3*3 필터 정의 (원래 이미지에 경계 강하게 표시)
fsharping2=np.array([[0.0, -1.0, 0.0],
                  [ -1.0, 5.0, -1.0],
                  [ 0.0, -1.0, 0.0]])

# result = cv.filter2D(gray, -1, faverage)
# result = cv.filter2D(gray, -1, fsharping)
result = cv.filter2D(gray, -1, fsharping2)
cv.imshow('result', result)

gray = cv.imread('coins.png', cv.IMREAD_GRAYSCALE)
# opencv에서 제공하는 bluring 함수
# average = cv.blur(gray, (3,3))
average = cv.blur(gray, (9,9)) # 필터를 키우면 더 blur효과 더 커짐
cv.imshow('result - average', average)

# opencv에서 제공하는 중간값 함수
median = cv.medianBlur(gray, 3)
cv.imshow('result - median', median)

# 가우시안 필터처럼 가운데에 해당하는 위치에 가중치를 많이 주고 그 외에 부분에 가중치 적게 주는 것처럼 필터 안에서도
# 해당되는 가중치의 값을 다르게 측정
# 값이 비슷한 애들은 smoothing된 결과 처리, edge처럼 서로 다른 값이 포함되어 있는 값은 smoothing 시키지 않음, 값을 무너트리지 않음
bilateral = cv.bilateralFilter(gray, -1, sigmaColor=5, sigmaSpace=5)
# 각 픽셀과 주변 요소들로부터 가중 평균을 구함 => 가우시안과 유사
# 단, 픽셀값의 차이도 같이 사용하여 유사한 픽셀에 더 큰 가중치를 두는 방법
# 경계선을 유지하며 스무딩
cv.imshow('result - bilateral', bilateral)

cv.waitKey()
cv.destroyAllWindows()