from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore						
from PyQt5.QtCore import QTimer,Qt
from PyQt5 import QtGui							
from tkinter import filedialog					
from tkinter import * 
from matplotlib import pyplot as plt 			
from matplotlib.widgets import Button
import sys									
import os										
import cv2										
import numpy as np 								
import winGuiAuto			
import win32gui
import win32con									
import keyboard								
import shutil									
							

def nothing(x):
	pass

image_x, image_y = 64,64						

from tensorflow.keras.models import load_model
classifier = load_model('ISLModel.h5')			


def fileSearch():
	fileEntry=[]
	for file in os.listdir("SampleGestures"):
	    if file.endswith(".png"):
	    	fileEntry.append(file)
	return fileEntry

def clearfunc(cam):
	cam.release()
	cv2.destroyAllWindows()

def clearfunc2(cam):
	cam.release()
	cv2.destroyAllWindows()

def controlTimer(self):
	self.timer.isActive()
	self.cam = cv2.VideoCapture(0)
	self.timer.start(20)


def predictor():
	import numpy as np
	from tensorflow.keras.preprocessing import image
	test_image = image.load_img('1.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	fileEntry=fileSearch()
	for i in range(len(fileEntry)):
		image_to_compare = cv2.imread("./SampleGestures/"+fileEntry[i])
		original = cv2.imread("1.png")
		sift = cv2.xfeatures2d.SIFT_create()
		kp_1, desc_1 = sift.detectAndCompute(original, None)
		kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
	
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		flann = cv2.FlannBasedMatcher(index_params, search_params)
	
		matches = flann.knnMatch(desc_1, desc_2, k=2)
	
		good_points = []
		ratio = 0.6
		for m, n in matches:
		      if m.distance < ratio*n.distance:
		             good_points.append(m)
		if(abs(len(good_points)+len(matches))>20):		
			gesname=fileEntry[i]
			gesname=gesname.replace('.png','')
			return gesname

	if result[0][0] == 1:
		return '0'
	elif result[0][1] == 1:
		return '1'
	elif result[0][2] == 1:
		return '2'
	elif result[0][3] == 1:
		return '3'
	elif result[0][4] == 1:
		return '4'
	elif result[0][5] == 1:
		return '5'
	elif result[0][6] == 1:
		return '6'
	elif result[0][7] == 1:
		return '7'
	elif result[0][8] == 1:
		return '8'
	elif result[0][9] == 1:
		return '9'
	elif result[0][10] == 1:
		return 'A'
	elif result[0][11] == 1:
		return 'B'
	elif result[0][12] == 1:
		return 'C'
	elif result[0][13] == 1:
		return 'CALL'
	elif result[0][14] == 1:
		return 'D'
	elif result[0][15] == 1:
		return 'E'
	elif result[0][16] == 1:
		return 'F'
	elif result[0][17] == 1:
		return 'G'
	elif result[0][18] == 1:
		return 'H'
	elif result[0][19] == 1:
		return 'HELP'
	elif result[0][20] == 1:
		return 'I'
	elif result[0][21] == 1:
		return 'J'
	elif result[0][22] == 1:
		return 'K'
	elif result[0][23] == 1:
		return 'L'
	elif result[0][24] == 1:
		return 'M'
	elif result[0][25] == 1:
		return 'N'
	elif result[0][26] == 1:
		return 'O'
	elif result[0][27] == 1:
		return 'P'
	elif result[0][28] == 1:
		return 'Q'
	elif result[0][29] == 1:
		return 'R'
	elif result[0][30] == 1:
		return 'S'
	elif result[0][31] == 1:
		return 'T'
	elif result[0][32] == 1:
		return 'THANK_YOU'
	elif result[0][33] == 1:
		return 'U'
	elif result[0][34] == 1:
		return 'V'
	elif result[0][35] == 1:
		return 'W'
	elif result[0][36] == 1:
		return 'X'
	elif result[0][37] == 1:
		return 'Y'
	elif result[0][38] == 1:
		return 'Z'
	

class Dashboard(QtWidgets.QMainWindow):
	def __init__(self):
		super(Dashboard, self).__init__()
		self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
		cv2.destroyAllWindows()
		self.setWindowIcon(QtGui.QIcon('icons/windowLogo.png'))
		self.title = 'Indian Sign language Recognition'
		uic.loadUi('UI_Files/dash.ui', self)
		self.setWindowTitle(self.title)
		self.timer = QTimer()
		if(self.scan_sinlge.clicked.connect(self.scanSingle)==True):
			self.timer.timeout.connect(self.scanSingle)
		self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.exit_button.clicked.connect(self.quitApplication)
		self._layout = self.layout()
		self.label_3 = QtWidgets.QLabel()
		movie = QtGui.QMovie("icons/isl.gif")
		self.label_3.setMovie(movie)
		self.label_3.setGeometry(0,160,780,441)
		movie.start()
		self._layout.addWidget(self.label_3)
		self.setObjectName('Message_Window')
		
	def quitApplication(self):
		userReply = QMessageBox.question(self, 'Quit Application', "Are you sure you want to quit this app?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if userReply == QMessageBox.Yes:
			keyboard.press_and_release('alt+F4')

	def scanSingle(self):
		try:
			clearfunc(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/scan_single.ui', self)
		self.setWindowTitle(self.title)
		if(self.scan_sinlge.clicked.connect(self.scanSingle)):
			controlTimer(self)
		self.pushButton_2.clicked.connect(lambda:clearfunc(self.cam))
		self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		try:
			self.exit_button.clicked.connect(lambda:clearfunc(self.cam))
		except:
			pass
		self.exit_button.clicked.connect(self.quitApplication)
		img_text = ''
		while True:
			ret, frame = self.cam.read()
			frame = cv2.flip(frame,1)
			try:
				frame=cv2.resize(frame,(321,270))
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				img1 = cv2.rectangle(frame, (150,50),(300,200), (0,255,0), thickness=2, lineType=8, shift=0)
			except:
				keyboard.press_and_release('esc')

			height1, width1, channel1 = img1.shape
			step1 = channel1 * width1
			qImg1 = QImage(img1.data, width1, height1, step1, QImage.Format_RGB888)
			try:
				self.label_3.setPixmap(QPixmap.fromImage(qImg1))
				slider1=self.trackbar.value()
			except:
				pass

			lower_blue = np.array([0, 0, 0])
			upper_blue = np.array([179, 255, slider1])

			imcrop = img1[52:198, 152:298]
			hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_blue, upper_blue)

			cv2.namedWindow("mask", cv2.WINDOW_NORMAL )
			cv2.imshow("mask", mask)
			cv2.setWindowProperty("mask",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			cv2.resizeWindow("mask",118,108)
			cv2.moveWindow("mask", 894,271)

			hwnd = winGuiAuto.findTopWindow("mask")
			win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

			try:
				self.textBrowser.setText("\n\n\t"+str(img_text))
			except:
				pass

			img_name = "1.png"
			save_img = cv2.resize(mask, (image_x, image_y))
			cv2.imwrite(img_name, save_img)
			img_text = predictor()

			if cv2.waitKey(1) == 27:
			    break

		self.cam.release()
		cv2.destroyAllWindows()

app = QtWidgets.QApplication([])
win = Dashboard()
win.show()
sys.exit(app.exec())