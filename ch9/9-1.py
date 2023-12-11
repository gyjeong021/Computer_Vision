import numpy as np
import cv2 as cv
import sys
from PIL import ImageFont, ImageDraw, Image

def construct_yolo_v3():
    # f=open('coco_names.txt', 'r')
    f=open('coco_names_kor.txt', 'r', encoding='UTF-8') # 한글 파일 읽기
    class_names=[line.strip() for line in f.readlines()] # 80개의 string을 가진 배열로 만듦

    model=cv.dnn.readNet('yolov3.weights','yolov3.cfg') # dnn의 readNet을 이용
    layer_names=model.getLayerNames() # 각각의 층에 대한 이름을 가져옴
    print(layer_names)
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()] # output 층 가져오기
    print(out_layers)

    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True) # yolo에 들어갈 수 있는 형태로 만들어줌
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers) # 결과를 3개 받음
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        print(len(output))
        for vec85 in output:
            scores=vec85[5:] # 인덱스 5부터 쭉 80개 가져오기
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                print(vec85)
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    # None Maximal Suppression Boxes: 신뢰도 0.5 이상, 박스끼리 겹치는 부분 0.4 이상이면 같은 물체로 간주
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()		# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔

# img=cv.imread('soccer.jpg')
img=cv.imread('busy_street.jpg')
if img is None: sys.exit('파일이 없습니다.')

res=yolo_detect(img,model,out_layers)	# YOLO 모델로 물체 검출

# 폰트 설정
font = ImageFont.truetype('fonts/gulim.ttc', 20)
# opencv 이미지를 pil로 옮김
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)

for i in range(len(res)):			# 검출된 물체를 영상에 표시
    x1,y1,x2,y2,confidence,id=res[i]
    text=str(class_names[id])+'%.3f'%confidence
    # cv.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
    # cv.putText(img, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

    # opencv에서 그냥 바로 한글 사용할 수 없음
    draw.rectangle((x1,y1,x2,y2), outline=tuple(colors[id].astype(int)),width=2)
    draw.text((x1,y1+30), text, font=font, fill=tuple(colors[id].astype(int)), width=2)

# pil로 되어있는 이미지를 다시 opencv에서 읽을 수 있는 형태로 변환
img = np.array(img_pil)
cv.imshow("Object detection by YOLO v.3",img)

cv.waitKey()
cv.destroyAllWindows()