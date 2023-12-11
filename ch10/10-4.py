import cv2 as cv
import mediapipe as mp

img=cv.imread('BSDS_376001.jpg')

mp_face_detection=mp.solutions.face_detection
mp_drawing=mp.solutions.drawing_utils # 검출된 결과 화면에 표현하는 tool

face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) # 설정
# 모델 인덱스는 0 또는 1이다.
# 0을 사용하면 카메라 2m 이내의 부분적 모델 촬영에 적합하고,
# 1은 5m 이내에서 전신 모델을 촬영하는데 적합
# 지정하지 않을 경우 기본값은 0
res=face_detection.process(cv.cvtColor(img,cv.COLOR_BGR2RGB)) # 실행

if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    print(len(res.detections))
    for detection in res.detections:
        print(detection)
        mp_drawing.draw_detection(img,detection)
    cv.imshow('Face detection by MediaPipe',img)

cv.waitKey()
cv.destroyAllWindows()