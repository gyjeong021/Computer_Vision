import cv2 as cv

# gray = cv.imread('soccer.jpg', cv.IMREAD_GRAYSCALE)
# cv.imshow('original - gray', gray)

# gray_mask = cv.inRange(gray, 120,170)    # 120~170이면 white, 아니면 black
# cv.imshow('inRange', gray_mask)

img = cv.imread('soccer.jpg')
cv.imshow('original', img)

# 사람이 느끼는 색상의 범위를 표현해줄 때는 RGB 값보다는 HSV가 좋음
# HSV는 명도와 채도를 표현하기 때문
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
red_mask = cv.inRange(hsv_img, (-10,50,50), (10,255,255)) # hsv에서 red의 범위
img_red = cv.bitwise_and(img, img, mask=red_mask)
# A and B : if B=0, 무조건 0 (A값 상관없이), else if B=1, A=0 -> 0, A=1 -> 1

# cv.imshow('red detection', red_mask)
cv.imshow('red detection', img_red)

cv.waitKey()
cv.destroyAllWindows()