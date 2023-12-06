from PyQt5.QtWidgets import *
import sys
import cv2 as cv
       
class Video(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')	# 윈도우 이름과 위치 지정
        self.setGeometry(200,200,500,100)

        videoButton=QPushButton('비디오 켜기',self)	# 버튼 생성
        captureButton=QPushButton('프레임 잡기',self)
        saveButton=QPushButton('프레임 저장',self)
        quitButton=QPushButton('나가기',self)
        
        videoButton.setGeometry(10,10,100,30)		# 버튼 위치와 크기 지정
        captureButton.setGeometry(110,10,100,30)
        saveButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(310,10,100,30)
        
        videoButton.clicked.connect(self.videoFunction) # 콜백 함수 지정
        captureButton.clicked.connect(self.captureFunction)         
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
       
    def videoFunction(self): # 비디오 파일을 가지고 와서 재생해 주는 함수
        # self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
        self.cap = cv.VideoCapture('../ch10/slow_traffic_small.mp4') # 동영상 파일
        if not self.cap.isOpened(): self.close()
            
        while True:
            ret,self.frame=self.cap.read() # ret은 videoFunction에서만 사용, cap이나 frame은 다른 메소드에서도 사용해서 self 붙여줌
            if not ret: break            
            cv.imshow('video display',self.frame)
            cv.waitKey(1)

    def captureFunction(self): # 캡처한 frame 저장하는 함수
        self.capturedFrame=self.frame
        cv.imshow('Captured Frame',self.capturedFrame)
        
    def saveFunction(self):				# 파일 저장
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./') # save할 수 있는 이름을 받아들임
        # QFileDialog : 파일 탐색기를 열어주는 함수
        cv.imwrite(fname[0],self.capturedFrame) # 첫번째 인자 : 저장할 파일 이름
        
    def quitFunction(self):
        self.cap.release()				# 카메라와 연결을 끊음
        cv.destroyAllWindows()
        self.close()

# 공통으로 들어가는 부분!!
app=QApplication(sys.argv) 
win=Video() 
win.show()
app.exec_()