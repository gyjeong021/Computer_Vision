import cv2 as cv
import keyboard

img = cv.imread('nct127.jpg')
#cv.imshow('Orginial',img)

color_model = None

def skin_detect():
    if color_model == 'YCbCr':
        ycbcr_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        ycbcr_mask = cv.inRange(ycbcr_img, (0, 77, 133), (255, 127, 173))  # ycbcr에서 피부색의 범위
        img_red = cv.bitwise_and(img, img, mask=ycbcr_mask)
        cv.imshow('red detection - YCbCr', img_red)
    elif color_model == 'HSV':
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv_mask = cv.inRange(hsv_img, (0, 70, 50), (50, 150, 255))  # hsv에서 피부색의 범위
        img_red = cv.bitwise_and(img, img, mask=hsv_mask)
        cv.imshow('red detection - HSV', img_red)

if keyboard.is_pressed('y'):
    color_model = 'YCbCr'
    skin_detect()

elif keyboard.is_pressed('h'):
    color_model = 'HSV'
    skin_detect()

cv.waitKey()
cv.destroyAllWindows()