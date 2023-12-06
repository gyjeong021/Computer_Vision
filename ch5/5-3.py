import cv2 as cv
import numpy as np
import time

# 2개의 이미지를 가져와 gray로 변환
img1=cv.imread('mot_color70.jpg')[190:350,440:560] # 버스를 크롭하여 모델 영상으로 사용
gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2=cv.imread('mot_color83.jpg')			     # 장면 영상
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)
print('특징점 개수:',len(kp1),len(kp2))
# // 5-2에서 한 내용 동일

start=time.time()

# 전수조사 방법 사용
# bf_matcher=cv.BFMatcher()
# knn_match=bf_matcher.knnMatch(des1,des2,2)
# knnMatch는 가장 유사한 특징점 k(=2)개를 결과로 받음

# 앞에서 계산한 기술자들을 매칭해주는 객체 생성
# 전수조사(Brute-Force)가 아닌 FLANN 방식 사용
flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match=flann_matcher.knnMatch(des1,des2,2) # k = 2, 가장 가까운 값, 두번째로 가까운 값
print(len(knn_match))

T=0.7
good_match=[]
for nearest1,nearest2 in knn_match:
    if (nearest1.distance/nearest2.distance)<T: # 매칭 전략 : 최근접 이웃 거리 비율
        good_match.append(nearest1)
print(len(good_match))
print(good_match[0].queryIdx, ' -- ', good_match[0].trainIdx, ' : ', good_match[0].distance)
print('매칭에 걸린 시간:',time.time()-start) 

img_match=np.empty((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3),dtype=np.uint8) # 배열을 만듦, 두개의 이미지를 하나의 이미지로 만들기 위해
# 두개의 이미지 중 큰 y 값, 두개의 이미지 더한 x 값, color
cv.drawMatches(img1,kp1,img2,kp2,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# drawMatches(첫번째 이미지, 첫번째 keypoint, 두번째 이미지, 두번째 keypoint, 둘 사이에서 찾은 매칭, output 이미지
# -> 매칭된 것 직선으로 연결해줌

cv.imshow('Good Matches', img_match)

k=cv.waitKey()
cv.destroyAllWindows()