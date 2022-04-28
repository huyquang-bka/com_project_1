# Common
import sys
import time

# QT
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore
from App_UI import Ui_MainWindow

# Detector thread
from Detector2 import DetectorThread

# Misc
import numpy as np
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.thread = {}

        self.uic.btn_start_cam1.clicked.connect(self.start_worker_1)
        self.uic.btn_start_cam2.clicked.connect(self.start_worker_2)

        self.uic.btn_stop_cam1.clicked.connect(self.stop_worker_1)
        self.uic.btn_stop_cam2.clicked.connect(self.stop_worker_2)

    def start_worker_1(self):
        self.thread[1] = DetectorThread(index=1)
        self.thread[1].setup(r"D:\cam_thu_vien\ch09_C5_ra_chiều_tối_06012022.mp4", 'yolov5s.pt')
        self.thread[1].start()
        self.thread[1].signal.connect(self.my_function)
        self.uic.btn_start_cam1.setEnabled(False)
        self.uic.btn_stop_cam1.setEnabled(True)

    def start_worker_2(self):
        self.thread[2] = DetectorThread(index=2)
        self.thread[2].setup(r"D:\cam_thu_vien\ch09_C5_ra_sáng_07012022.mp4", 'yolov5s.pt')
        self.thread[2].start()
        self.thread[2].signal.connect(self.my_function)
        self.uic.btn_start_cam2.setEnabled(False)
        self.uic.btn_stop_cam2.setEnabled(True)

    def stop_worker_1(self):
        self.thread[1].stop()
        self.uic.btn_stop_cam1.setEnabled(False)
        self.uic.btn_start_cam1.setEnabled(True)

    def stop_worker_2(self):
        self.thread[2].stop()
        self.uic.btn_stop_cam2.setEnabled(False)
        self.uic.btn_start_cam2.setEnabled(True)

    def my_function(self, img):
        img_c = img
        rgb_img = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        qt_img = QtGui.QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QtGui.QImage.Format_RGB888)
        thread = self.sender().index

        if thread == 1:
            self.uic.img1.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img1.width(), self.uic.img1.height()))
        if thread == 2:
            self.uic.img2.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img2.width(), self.uic.img2.height()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
