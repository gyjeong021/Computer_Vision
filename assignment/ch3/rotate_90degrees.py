import cv2 as cv
import numpy as np

img=cv.imread('ocean.jpg')

rows,cols = img.shape[:2] # shape은 영상의 크기(세로,가로)

def rotate(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭했을 때 시계반대방향 90도 회전
        src_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0]])
        dst_points = np.float32([[0, cols - 1], [rows - 1, cols - 1], [0, 0]])
    elif event == cv.EVENT_RBUTTONDOWN:  # 마우스 오른쪽 버튼 클릭했을 때 시계방향 90도 회전
        src_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0]])
        dst_points = np.float32([[0, cols - 1], [0, 0], [rows - 1, cols - 1]])

    affine_matrix = cv.getAffineTransform(src_points, dst_points)  # 변환 행렬 만듦
    img_symmetry = cv.warpAffine(img, affine_matrix, (cols, rows))  # 3번째 인자는 출력영상의 크기(가로,세로)

    cv.imshow('rotate 90', img_symmetry)

cv.namedWindow('rotate 90')
cv.imshow('rotate 90',img)

cv.setMouseCallback('rotate 90',rotate)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
