import cv2 as cv 

img=cv.imread('apples.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# HoughCircles 자체적으로 canny 처리함, 함수 내에서 엣지를 구함
# 그래서 엣지 이미지 넣지 않아도 됨, 그레이 영상 넣으면 됨
# method : 검출방법, HOUGH_GRADIENT: 엣지 값 이용
# dp : 이미지 해상도 : accumulator 해상도, 1이면 두 해상도 같음, 누적배열과 영상의 비율 같음
# dist : 검출된 원 중심 사이의 최소 거리, 원이 한 곳에 몰려있지 않도록 설정
# param1 : canny의 높은 threshold
# param2 : 누적 threshold, 주변보다는 커도 일정값 이상은 나오도록 정해주는 값
# minRadius, max Radius : 검출할 원 반지름 범위 (클수록 실행 시간과 사용되는 메모리 커짐)
# 반지름에 따라서 이차원 배열에 추가할 수 있음
apples=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,200,param1=150,param2=80,minRadius=50,maxRadius=120)

if apples is not None: # param2=80이여도 오류 발생하지 않음
    for i in apples[0]:
        print(i) # 3번째 인자 50~120 사이
        cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)

cv.imshow('Apple detection',img)  

cv.waitKey()
cv.destroyAllWindows()