import cv2 as cv 
import sys
import numpy as np

img = cv.imread('road.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 2, 2)
canny = cv.Canny(blur, 100, 200, 5) # 엣지 이미지 구하기
# canny 알고리즘은 4단계로 구성
# 첫번째 단계에서 노이즈 제거하는 가우시안 블러 처리를 진행
# 따라서 그레이 영상 처리 후 바로 canny 처리해도 괜찮
# 추가적으로 노이즈 제거가 필요한 경우에 더 추가 가능

# np.pi = 180
rho, theta = 1,  np.pi / 180
# HoughCircle는 함수 내에서 canny 알고리즘을 돌려 따로 canny를 하지않고 gray 영상을 넣지만
# HoughLinesP는 HoughCircle과 달리 엣지 이미지를 입력으로 받음
lines = cv.HoughLinesP(canny, rho, theta, 10, minLineLength=20, maxLineGap=5)
# 주로 에지영상을 입력
# 직선으로 판단할 threshold

if lines is not None:
    for line in lines:    # 검출된 선 그리기
        print(line)
        x1, y1, x2, y2 = line[0]
        len = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        print(len)
        cv.line(img, (x1,y1), (x2, y2), (0,255,0), 2)

cv.imshow("image", img)

cv.waitKey()
cv.destroyAllWindows()