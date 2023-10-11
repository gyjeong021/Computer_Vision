import cv2 as cv

# img=cv.imread('soccer.jpg')
img=cv.imread('check.png')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray = cv.blur(gray, (3,3)) # 잡음 제거 효과

# 첫번째 인자 : 이미지(그레이 이미지만 가능)
# 두번째 인자 : 데이터 타입 // 그레이=1byte에 저장
# but 소벨은 음수의 값도 나올 수 있어 overflow 발생할 수 있음 (CV_32F=4bytes=32bit)
# 3,4번쨰 인자,왼쪽 오른쪽 빼기(1,0), 위아래 빼기 구분(0,1)
# 5번째 인자 : 필터 사이즈
# grad_x=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)	# 소벨 연산자 적용, sobel_x
# grad_y=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3) # sobel_y
grad = cv.Laplacian(gray, cv.CV_32F)

# sobel_x=cv.convertScaleAbs(grad_x)	# 절대값을 취해 양수 영상으로 변환
# sobel_y=cv.convertScaleAbs(grad_y)
lap=cv.convertScaleAbs(grad)

# addWeighted : 행렬 더하기 (alpha+beta=1이 되도록 설정)
# edge_strength=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)	# 에지 강도 계산
# 0.5*sobel_x + 0.5*sobel_y + 0,0

cv.imshow('Original',gray)
# cv.imshow('sobelx',sobel_x) # 왼쪽 오른쪽, 수직 경계선 뚜렷
# cv.imshow('sobely',sobel_y) # 위 아래, 수평 경계선 뚜렷
# cv.imshow('edge strength',edge_strength)
cv.imshow('laplacian',lap)

cv.waitKey()
cv.destroyAllWindows()

# 특정 방향에 대한 edge 값을 구하고 싶으면 sobel이나 prewitt 사용
# 가로, 세로 방향 상관없이 주변과 차이 크게 나면 찾겠다 -> 라플라시안 사용