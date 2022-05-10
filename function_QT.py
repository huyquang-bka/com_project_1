# Common
import sys
import time

# QT
import torch
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from ui_display import Ui_MainWindow
# Detector thread
from Detector2 import DetectorThread, PostApi, DetectorFace, DetectorFree
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        # self.cap = cv2.VideoCapture()
        self.thread = {}
        self.post_api_thread = {}
        self.uic.btn_add_cam.clicked.connect(self.add_CAM)
        self.uic.btn_delete_cam.clicked.connect(self.del_CAM)
        self.uic.btn_start_1.clicked.connect(self.start_worker_1)
        self.uic.btn_start_2.clicked.connect(self.start_worker_2)
        self.uic.btn_start_3.clicked.connect(self.start_worker_3)
        self.uic.btn_start_4.clicked.connect(self.start_worker_4)

        self.uic.btn_stop_1.clicked.connect(self.stop_worker_1)
        self.uic.btn_stop_2.clicked.connect(self.stop_worker_2)
        self.uic.btn_stop_3.clicked.connect(self.stop_worker_3)
        self.uic.btn_stop_4.clicked.connect(self.stop_worker_4)

    def start_worker_1(self):
        self.thread[1] = DetectorThread(index=1)
        self.post_api_thread[1] = PostApi(index=1)
        rstp1 = self.uic.list_cam1.currentText()
        self.thread[1].setup(rstp1, 'yolov5s.pt')
        self.thread[1].start()
        self.post_api_thread[1].start()
        self.thread[1].signal.connect(self.my_function)
        self.uic.btn_start_1.setEnabled(False)
        self.uic.btn_stop_1.setEnabled(True)

    def start_worker_2(self):
        self.thread[2] = DetectorFace(index=2)
        self.post_api_thread[2] = PostApi(index=2)
        rstp2 = self.uic.list_cam2.currentText()
        self.thread[2].setup(rstp2)
        self.thread[2].start()
        self.post_api_thread[2].start()
        self.thread[2].signal.connect(self.my_function)
        self.uic.btn_start_2.setEnabled(False)
        self.uic.btn_stop_2.setEnabled(True)

    def start_worker_3(self):
        self.thread[3] = DetectorFree(index=3)
        rstp3 = self.uic.list_cam3.currentText()
        self.thread[3].setup(rstp3)
        self.thread[3].start()
        self.thread[3].signal.connect(self.my_function)
        self.uic.btn_start_3.setEnabled(False)
        self.uic.btn_stop_3.setEnabled(True)

    def start_worker_4(self):
        self.thread[4] = DetectorThread(index=4)
        rstp4 = self.uic.list_cam4.currentText()
        self.thread[4].setup(rstp4, 'yolov5s.pt')
        self.thread[4].start()
        self.thread[4].signal.connect(self.my_function)
        self.uic.btn_start_4.setEnabled(False)
        self.uic.btn_stop_4.setEnabled(True)

    def stop_worker_1(self):
        self.thread[1].stop()
        self.post_api_thread[1].stop()
        self.uic.btn_stop_1.setEnabled(False)
        self.uic.btn_start_1.setEnabled(True)

    def stop_worker_2(self):
        self.thread[2].stop()
        self.post_api_thread[2].stop()
        self.uic.btn_stop_2.setEnabled(False)
        self.uic.btn_start_2.setEnabled(True)

    def stop_worker_3(self):
        self.thread[3].stop()
        self.uic.btn_stop_3.setEnabled(False)
        self.uic.btn_start_3.setEnabled(True)

    def stop_worker_4(self):
        self.thread[4].stop()
        self.uic.btn_stop_4.setEnabled(False)
        self.uic.btn_start_4.setEnabled(True)

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
        if thread == 3:
            self.uic.img3.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img3.width(), self.uic.img3.height()))
        if thread == 4:
            self.uic.img4.setPixmap(
                QtGui.QPixmap.fromImage(qt_img).scaled(self.uic.img4.width(), self.uic.img4.height()))

    def add_CAM(self):
        _translate = QtCore.QCoreApplication.translate
        rstp = self.uic.text_rstp.toPlainText()
        self.uic.list_cam1.addItem(rstp)
        self.uic.list_cam2.addItem(rstp)
        self.uic.list_cam3.addItem(rstp)
        self.uic.list_cam4.addItem(rstp)
        self.uic.text_rstp.clear()

    def del_CAM(self):
        index = self.uic.text_rstp.currentIndex()
        self.uic.list_cam1.removeItem(index)
        self.uic.list_cam2.removeItem(index)
        self.uic.list_cam3.removeItem(index)
        self.uic.list_cam4.removeItem(index)
        self.uic.text_rstp.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
