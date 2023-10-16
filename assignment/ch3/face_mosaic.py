import cv2 as cv
import cvlib as cvl      # cvlib, tensorflow 설치
# cvlib는 파이썬에서 얼굴, 객체 인식을 위한 사용하기 쉬운 라이브러리
# opencv와 tensorflow를 사용하고 있기 때문에, cvlib와 함께 설치해야 함

ksize = 31              # 블러 처리에 사용할 커널 크기

img = cv.imread('nct127.jpg')
img_small=cv.resize(img,dsize=(0,0),fx=0.5,fy=0.5) # 반으로 축소

faces, confidences = cvl.detect_face(img_small)

for (x,y, x2,y2), conf in zip(faces, confidences):
    cv.rectangle(img_small, (x, y), (x2, y2), (0, 255, 0), 2)

#cv.imshow('face detection', img_small)

# 모자이크 처리 함수 정의
def apply_blur(image, factor=0.2):
    h, w, _ = image.shape
    small = cv.resize(image, (0, 0), fx=factor, fy=factor)
    return cv.resize(small, (w, h), interpolation=cv.INTER_LINEAR)

# 얼굴 부분에 모자이크 적용
for (x, y, x2, y2), _ in zip(faces, confidences):
    # 얼굴 좌표로 이미지를 자름
    face_roi = img_small[y:y2, x:x2]
    # 자른 부분에 모자이크 처리를 적용
    blurred_face = apply_blur(face_roi)
    # 모자이크 처리된 이미지를 원본 이미지에 복사
    img_small[y:y2, x:x2] = blurred_face

# 모자이크 처리된 얼굴 표시
cv.imshow("Blurred Face", img_small)
cv.waitKey(0)
cv.destroyAllWindows()

