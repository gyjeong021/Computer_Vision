import cv2 as cv
import numpy as np
img=cv.imread('rose.png')

rows,cols = img.shape[:2] # shape은 영상의 크기(세로,가로)

# Horizontal
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])

# Vertical
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,rows-1], [cols-1,rows-1], [0,0]])

# Origin
src_points = np.float32([[0,0], [0,rows-1], [cols-1,0]])
dst_points = np.float32([[cols-1,rows-1], [cols-1,0], [0,rows-1]])

affine_matrix = cv.getAffineTransform(src_points, dst_points) # 변환 행렬 만듦
img_symmetry = cv.warpAffine(img, affine_matrix, (cols,rows)) # 3번째 인자는 출력영상의 크기(가로,세로)

cv.imshow('Original',img)
cv.imshow('Symmetry',img_symmetry)

cv.waitKey()
cv.destroyAllWindows()