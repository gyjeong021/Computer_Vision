import cv2 as cv

img = cv.imread('nct127.jpg')
img_small=cv.resize(img,dsize=(0,0),fx=0.5,fy=0.5) # 반으로 축소
cv.imshow('Orginial',img_small)

color_model = None

def skin_detect():
    if color_model == 'YCbCr':
        ycbcr_img = cv.cvtColor(img_small, cv.COLOR_BGR2YCrCb)
        ycbcr_mask = cv.inRange(ycbcr_img, (0, 77, 133), (255, 127, 173))  # ycbcr에서 피부색의 범위
        img_red = cv.bitwise_and(img_small, img_small, mask=ycbcr_mask)
        cv.imshow('red detection - YCbCr', img_red)
    elif color_model == 'HSV':
        hsv_img = cv.cvtColor(img_small, cv.COLOR_BGR2HSV)
        hsv_mask = cv.inRange(hsv_img, (0, 70, 50), (50, 150, 255))  # hsv에서 피부색의 범위
        img_red = cv.bitwise_and(img_small, img_small, mask=hsv_mask)
        cv.imshow('red detection - HSV', img_red)

while True:
    key = cv.waitKey(1)  # 1밀리초 동안 키보드 입력 기다림
    if key==ord('y'):
        color_model = 'YCbCr'
        skin_detect()
    elif key==ord('h'):
        color_model = 'HSV'
        skin_detect()
    elif key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break

cv.destroyAllWindows()