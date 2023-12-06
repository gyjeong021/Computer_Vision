import cv2 as cv
import numpy as np
import sys

def onCornerHarris(thresh):
    CN = cv.normalize(C, 0, 400, cv.NORM_MINMAX)  # 트랙바는 정수만 다룸, 그래서 normalize 이용해서 소수인 숫자 키워줌
    rcorners = []
    for j in range(1, C.shape[0] - 1):  # 비최대 억제
        for i in range(1, C.shape[1] - 1):
            if CN[j, i] > thresh and sum(
                sum(CN[j, i] > CN[j - 1:j + 2, i - 1:i + 2])) == 8:  # 주변 8개와 비교해서 [ji]가 더 큰 값이 8이면, 모든 이웃보다 큰값을 가지면
                rcorners.append((i, j))

    for pt in rcorners:
        cv.circle(img, pt, 3, (255, 0, 255), -1)  # 좌표 표시
    print("임계값: %2d , 코너 개수: %2d" % (thresh, len(rcorners)))
    cv.imshow("harris detect", img)


img = cv.imread('shapes3.png', cv.IMREAD_COLOR)
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

median = cv.medianBlur(img, 3) # 잡음 제거를 위해 미디언 블러 사용 (이미지에 따라 선택 사항)
gray = cv.cvtColor(median, cv.COLOR_BGR2GRAY) # 코너 해리스의 입력 영상은 gray 이미지여야 함

blockSize = 4  # 이웃 화소 범위
apertureSize = 3  # 소벨 마스크 크기
k = 0.04  # 0.04~0.06 사이의 값 사용
thresh = 5  # 코너 응답 임계값
C = cv.cornerHarris(gray, blockSize, apertureSize, k)  # OpenCV 제공 함수

onCornerHarris(thresh)
cv.createTrackbar("Threshold", "harris detect", thresh, 30, onCornerHarris)

cv.waitKey(0)
cv.destroyAllWindows()