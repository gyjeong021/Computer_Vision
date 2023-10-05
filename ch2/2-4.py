import cv2 as cv
import sys

# cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
cap=cv.VideoCapture("../ch10/slow_traffic_small.mp4")

if not cap.isOpened():
    sys.exit('카메라 연결 실패')
    
while True: # 카메라를 종료하거나 동영상 프레임이 끝나면 False 되면서 while문 종료
    ret,frame=cap.read()			# 비디오를 구성하는 프레임 획득
    
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_small = cv.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)
    cv.imshow('Video display',gray_small)
    
    key=cv.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림, 숫자 없으면 다른 움직임이 있을 때까지 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감, ord() : 아스키 코드로 변환
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv.destroyAllWindows()