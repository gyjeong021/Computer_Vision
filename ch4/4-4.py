import cv2 as cv 

img=cv.imread('apples.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# HoughCircles 자체적으로 canny 처리함, 그래서 엣지 이미지 넣지 않아도 됨, 그레이 영상 넣으면 됨
# 첫번째 인자 : 입력 영상 (그레이 타입이여야 함)
# 두번째 인자 : 검출하는 방법 (cv.HOUGH_GRADIENT:엣지 값 이용)
# 세번째 인자 : 1 - 누적배열과 영상의 비율이 동일 (1:1)
# 네번째 인자 : circle과 circle 사이의 거리 (원이 한 곳에 몰려있지 않도록 설정)
# 다섯번째 인자 : 캐니 엣지에서 높은 쪽 = Thigh
# 여섯번째 인자 : 누적에서 주변보다는 커도 일정값 이상은 나오도록 정해주는 값
# 일곱,여덟번째 인자 : 반지름에 따라서 이차원 배열에 추가할 수 있는데 그때 찾으려는 원의 반지름 범위 (클수록 실행 시간과 사용되는 메모리 커짐)
apples=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,200,param1=150,param2=40,minRadius=50,maxRadius=120)

for i in apples[0]: 
    cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)

cv.imshow('Apple detection',img)  

cv.waitKey()
cv.destroyAllWindows()