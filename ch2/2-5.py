import cv2 as cv
import numpy as np
import sys

# cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
cap=cv.VideoCapture("../ch10/slow_traffic_small.mp4")

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

frames=[]
while True:
    ret,frame=cap.read()			# 비디오를 구성하는 프레임 획득
    
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
        
    cv.imshow('Video display',frame)
    
    key=cv.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림
    if key==ord('c'):	# 'c' 키가 들어오면 프레임을 리스트에 추가
        frames.append(frame) 
    elif key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv.destroyAllWindows() 

print(frames[0].shape) # 배열의 크기는 shape 이용해서 알 수 있음 (360, 640, 3)

if len(frames)>0:		# 수집된 영상이 있으면
    imgs=frames[0]
    for i in range(1,min(3,len(frames))):	# 최대 3개까지 이어 붙임
        imgs=np.hstack((imgs,frames[i])) # 가로 결합
        # imgs = np.vstack((imgs, frames[i])) # 세로 결합
    
    cv.imshow('collected images',imgs)

    print(imgs.shape) # (360, 1920, 3) 가로로 합쳐져 3배 더 길어짐

    cv.waitKey()
    cv.destroyAllWindows()