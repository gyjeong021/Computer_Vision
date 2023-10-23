import cv2 as cv
import numpy as np

# Image loading
img = cv.imread("water_coins.jpg")

# image grayscale conversion
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Threshold Processing
# 오츄 알고리즘 이용하여 이진화
ret, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("binarized image", bin_img)

# 이진화 후에 노이즈 제거는 블러링 작업보다는 모폴로지-openig(침식->팽창) 사용
# white(object) noise removal
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
b_opening = cv.dilate(cv.erode(bin_img,kernel,iterations=2),kernel,iterations=2)	# 열기
cv.imshow("white noise removal", b_opening)

# sure background area
# 확실한 배경 알기 위해 팽창시킴
sure_bg = cv.dilate(b_opening,kernel,iterations=3)
cv.imshow("sure background area", sure_bg)

# sure foreground area
# 확실한 물체를 찾기 위해 white 픽셀과 black 픽셀의 거리 측정
# distanceTransform : Binary 이미지에서  픽셀값이 0인 배경으로부터의 거리를 픽셀값이 255인 영역에 표현하는 방법
dist_transform = cv.distanceTransform(b_opening,cv.DIST_L2,5)
# 2 : 거리 공식
# 3 : 마스크 크기
dist = dist_transform * 10
dist = np.uint8(dist_transform) # 한 바이트로 표현
# cv.imshow("dist_transform", dist)

ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv.imshow("sure foreground area", sure_fg)

# unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg) # 확실한 배경에서 확실한 물체인 부분 빼기
cv.imshow("unknown region", unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# connectedComponents : 처음 물체인 픽셀 만나면 4방향 또는 8방향으로 연결해 한 물체로 봄
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers) # markers : seed 정보, 앞은 seed를 구하는 과정
img[markers == -1] = [255,0,0] # 두 영역 만나는 지점 = -1로
cv.imshow("watershed results", img)

cv.waitKey()
cv.destroyAllWindows()