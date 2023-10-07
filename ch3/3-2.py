import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('soccer.jpg')

# opencv에서는 컬러가 b, g, r 순으로 저장
# calcHist에서 반환되는 값은 각각 색깔에 대한 카운트 값
h=cv.calcHist([img],[2],None,[256],[0,256]) # 2번 채널인 R 채널에서 히스토그램 구함
plt.plot(h,color='r',linewidth=1) # linewidth : 선 두께

hg=cv.calcHist([img],[1],None,[256],[0,256]) # 1번 채널인 G 채널에서 히스토그램 구함
plt.plot(hg,color='g',linewidth=2, linestyle="dotted")

hb=cv.calcHist([img],[0],None,[256],[0,256]) # 번 채널인 B 채널에서 히스토그램 구함
plt.plot(hb,color='b',linewidth=3, linestyle="dashed")

plt.show()