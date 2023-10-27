import cv2 as cv 
import numpy as np
import sys
from PyQt5.QtWidgets import *
      
class Orim(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200,200,700,200)
       
        fileButton=QPushButton('파일',self) # 버튼 선언
        paintButton=QPushButton('페인팅',self)
        cutButton=QPushButton('오림',self)
        incButton=QPushButton('+',self)
        decButton=QPushButton('-',self)
        resetButton=QPushButton('초기화',self)
        saveButton=QPushButton('저장',self)
        quitButton=QPushButton('나가기',self)
        
        fileButton.setGeometry(10,10,100,30) # 윈도우에 버튼 배치
        paintButton.setGeometry(110,10,100,30)
        cutButton.setGeometry(210,10,100,30)
        incButton.setGeometry(310,10,50,30)
        decButton.setGeometry(360,10,50,30)
        resetButton.setGeometry(410,10,100,30) # 추가 : 초기화 버튼
        saveButton.setGeometry(510,10,100,30)
        quitButton.setGeometry(610,10,100,30)
        
        fileButton.clicked.connect(self.fileOpenFunction) # 콜백 함수
        paintButton.clicked.connect(self.paintFunction) 
        cutButton.clicked.connect(self.cutFunction)    
        incButton.clicked.connect(self.incFunction)              
        decButton.clicked.connect(self.decFunction)
        resetButton.clicked.connect(self.resetFuction)
        saveButton.clicked.connect(self.saveFunction)                         
        quitButton.clicked.connect(self.quitFunction)

        # 전역변수에 대한 선언도 init에서 함
        self.BrushSiz=5			# 페인팅 붓의 크기
        self.LColor,self.RColor=(255,0,0),(0,0,255) # 파란색 물체, 빨간색 배경

    # 8개의 콜백함수 : 버튼에 대한 콜백함수 + 마우스에 대한 콜백함수
    def fileOpenFunction(self):
        fname=QFileDialog.getOpenFileName(self,'Open file','./')
        self.img=cv.imread(fname[0]) # 인자에 파일 이름이 들어갔음, QFileDialog를 사용하면 하나의 이미지가 아닌 어떤 이미지든 사용 가능
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')  
        
        self.img_show=np.copy(self.img)	# 표시용 영상 
        cv.imshow('Painting',self.img_show)
        
        self.mask=np.zeros((self.img.shape[0],self.img.shape[1]),np.uint8) 
        self.mask[:,:]=cv.GC_PR_BGD	# 모든 화소를 배경일 것 같음으로 초기화


    def resetFuction(self):
        self.img_show = np.copy(self.img)  # 표시용 영상
        cv.imshow('Painting', self.img_show)

        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:, :] = cv.GC_PR_BGD  # 모든 화소를 배경일 것 같음으로 초기화

    def paintFunction(self):
        cv.setMouseCallback('Painting',self.painting) 
        
    def painting(self,event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN:   
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1) # 왼쪽 버튼을 클릭하면 파란색
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event==cv.EVENT_RBUTTONDOWN: 
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1) # 오른쪽 버튼을 클릭하면 빨간색
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)
        elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1) # 왼쪽 버튼을 클릭하고 이동하면 파란색
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1) # 오른쪽 버튼을 클릭하고 이동하면 빨간색 
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)
    
        cv.imshow('Painting',self.img_show)        
        
    def cutFunction(self): # grabcut 하기 위한 단계
        if cv.GC_FGD not in self.mask :
            return
        background=np.zeros((1,65),np.float64) 
        foreground=np.zeros((1,65),np.float64) 
        cv.grabCut(self.img,self.mask,None,background,foreground,5,cv.GC_INIT_WITH_MASK)
        mask2=np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
        self.grabImg=self.img*mask2[:,:,np.newaxis]
        cv.imshow('Scissoring',self.grabImg) 
        
    def incFunction(self):
        self.BrushSiz=min(20,self.BrushSiz+1) 
        
    def decFunction(self):
        self.BrushSiz=max(1,self.BrushSiz-1) 
        
    def saveFunction(self):
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./')
        cv.imwrite(fname[0],self.grabImg)
                
    def quitFunction(self):
        cv.destroyAllWindows()        
        self.close()

# 선언된 클래스를 Qt에서 실행해주는, 항상 포함되는 동일한 구조
app=QApplication(sys.argv) 
win=Orim() 
win.show()
app.exec_()