import cv2 as cv
import sys

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 동영상을 가져오는 클래스
cap = cv.VideoCapture('face2.mp4')

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:  # 무한루프로
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득(frame)

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red_mask = cv.inRange(hsv_img, (0, 10, 60), (20, 255, 255))  # hsv에서 skin color의 범위
    img_red = cv.bitwise_and(frame, frame, mask=red_mask)

    cv.imshow('skin color detection', img_red)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
