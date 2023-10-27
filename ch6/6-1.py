from PyQt5.QtWidgets import *
import sys
import winsound

class BeepSound(QMainWindow):
    def __init__(self) : # 초기화 했을 때 제일 먼저 동작
        super().__init__()
        self.setWindowTitle('삑 소리 내기') 		# 윈도우 이름과 위치 지정
        self.setGeometry(200,200,500,100) # 좌측 상단 기준으로 (x,y,w,h)

        shortBeepButton=QPushButton('짧게 삑',self)	# 버튼 생성
        longBeepButton=QPushButton('길게 삑',self)
        quitButton=QPushButton('나가기',self)
        self.label=QLabel('환영합니다!',self) # __init__ 함수 아닌 다른 함수에서도 사용하면 self. 앞에 붙여야함
        
        shortBeepButton.setGeometry(10,10,100,30)	# 버튼 위치와 크기 지정, 좌측 상단 기준으로 10*10 위치에 100*30 크기로
        longBeepButton.setGeometry(110,10,100,30)   # setGeometry 사용해야 버튼이 윈도우에 들어감
        quitButton.setGeometry(210,10,100,30)
        self.label.setGeometry(10,40,500,70)
        
        shortBeepButton.clicked.connect(self.shortBeepFunction) # 클릭되었을 때 콜백 함수 지정
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)
       
    def shortBeepFunction(self):
        self.label.setText('주파수 1000으로 0.5초 동안 삑 소리를 냅니다.')   
        winsound.Beep(1000,500) # 프로그램에서 시간 단위는 ms, 1000이 1초
        
    def longBeepFunction(self):
        self.label.setText('주파수 1000으로 3초 동안 삑 소리를 냅니다.')        
        winsound.Beep(1000,3000) 
                
    def quitFunction(self):
        self.close()

# 이 4줄 꼭 반드시 포함되어야 함
app=QApplication(sys.argv) # QApplication으로 pyQt 생성
win=BeepSound() # pyQt에 대해서 동작할 윈도우 클래스 생성
win.show()
app.exec_() # 무한루프로 사용자에 대한 응답 처리