import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# 이미지 불러오기
image = cv2.imread('nct127.jpg')

# 얼굴 검출
faces, confidences = cv.detect_face(image)

# 검출된 얼굴 주위에 박스 그리기
image_with_boxes = draw_bbox(image, faces)

# 얼굴이 그려진 이미지 표시
cv2.imshow("Face Detection", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 검출된 얼굴 영역 추출
for face in faces:
    x, y, w, h = face
    face_region = image[y:y+h, x:x+w]

# 추출된 얼굴 부분 표시
cv2.imshow("Extracted Face", face_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 모자이크 처리 함수 정의
def apply_blur(image, factor=0.2):
    w, h, _ = image.shape
    small = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

# 얼굴 부분에 모자이크 적용
blurred_face = apply_blur(face_region)

# 모자이크 처리된 얼굴 표시
cv2.imshow("Blurred Face", blurred_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 모자이크 처리된 얼굴을 원래 이미지에 적용
image[y:y+h, x:x+w] = blurred_face

# 모자이크 처리된 이미지 표시
cv2.imshow("Image with Blurred Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
