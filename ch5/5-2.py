import cv2 as cv

img=cv.imread('mot_color70.jpg') # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 특징 검출 알고리즘은 대부분 입력 영상이 gray

sift=cv.SIFT_create() # sift 계산할 수 있는 객체 생성
kp,des=sift.detectAndCompute(gray,None)
# sift.detectAndCompute(inputImg,mask=None)
# 특징점 검출과 기술자 계산을 한 번에 수행
# kp = keypoint(해당되는 특징점의 위치), des = descriptor(그것을 기술한 것)
# 2mask : 특징점 검출에 사용할 필터

print(len(kp))
print(kp[0].pt, kp[0].size, kp[0].octave, kp[0].angle) # 위치, 사이즈, 옥타브, 각도
print(des[0]) # 총 128개의 값 기술, 16개 block에 대한 히스토그램 값

# gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 스케일에 의해 검출된 크기, 방향에 대한 정보 포함
gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DEFAULT) # 특징 위치만 표시
# drawKeypoints(inputImg,kp,outImg=None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# flags : cv.DRAW_MATCHES_FLAGS_DEFAULT, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINT, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
cv.imshow('sift', gray)

k=cv.waitKey()
cv.destroyAllWindows()