# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_display.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.link_rstp = QtWidgets.QLabel(self.centralwidget)
        self.link_rstp.setGeometry(QtCore.QRect(20, 10, 41, 31))
        self.link_rstp.setObjectName("link_rstp")
        self.text_rstp = QtWidgets.QTextEdit(self.centralwidget)
        self.text_rstp.setGeometry(QtCore.QRect(60, 10, 271, 31))
        self.text_rstp.setObjectName("text_rstp")
        self.btn_add_cam = QtWidgets.QPushButton(self.centralwidget)
        self.btn_add_cam.setGeometry(QtCore.QRect(350, 10, 89, 31))
        self.btn_add_cam.setObjectName("btn_add_cam")
        self.btn_delete_cam = QtWidgets.QPushButton(self.centralwidget)
        self.btn_delete_cam.setGeometry(QtCore.QRect(450, 10, 101, 31))
        self.btn_delete_cam.setObjectName("btn_delete_cam")
        self.list_cam1 = QtWidgets.QComboBox(self.centralwidget)
        self.list_cam1.setGeometry(QtCore.QRect(60, 80, 91, 25))
        self.list_cam1.setObjectName("list_cam1")
        self.list_cam1.addItem("")
        self.list_cam1.addItem("")
        self.btn_start_1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_1.setGeometry(QtCore.QRect(170, 80, 89, 25))
        self.btn_start_1.setObjectName("btn_start_1")
        self.btn_start_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_2.setGeometry(QtCore.QRect(740, 80, 89, 25))
        self.btn_start_2.setObjectName("btn_start_2")
        self.list_cam2 = QtWidgets.QComboBox(self.centralwidget)
        self.list_cam2.setGeometry(QtCore.QRect(630, 80, 91, 25))
        self.list_cam2.setObjectName("list_cam2")
        self.list_cam2.addItem("")
        self.list_cam2.addItem("")
        self.img1 = QtWidgets.QLabel(self.centralwidget)
        self.img1.setGeometry(QtCore.QRect(60, 130, 331, 231))
        self.img1.setObjectName("img1")
        self.img2 = QtWidgets.QLabel(self.centralwidget)
        self.img2.setGeometry(QtCore.QRect(550, 130, 331, 231))
        self.img2.setObjectName("img2")
        self.btn_start_3 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_3.setGeometry(QtCore.QRect(180, 380, 89, 25))
        self.btn_start_3.setObjectName("btn_start_3")
        self.list_cam3 = QtWidgets.QComboBox(self.centralwidget)
        self.list_cam3.setGeometry(QtCore.QRect(70, 380, 91, 25))
        self.list_cam3.setObjectName("list_cam3")
        self.list_cam3.addItem("")
        self.list_cam3.addItem("")
        self.list_cam4 = QtWidgets.QComboBox(self.centralwidget)
        self.list_cam4.setGeometry(QtCore.QRect(630, 380, 91, 25))
        self.list_cam4.setObjectName("list_cam4")
        self.list_cam4.addItem("")
        self.list_cam4.addItem("")
        self.btn_start_4 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_4.setGeometry(QtCore.QRect(740, 380, 89, 25))
        self.btn_start_4.setObjectName("btn_start_4")
        self.img3 = QtWidgets.QLabel(self.centralwidget)
        self.img3.setGeometry(QtCore.QRect(60, 430, 331, 231))
        self.img3.setObjectName("img3")
        self.img4 = QtWidgets.QLabel(self.centralwidget)
        self.img4.setGeometry(QtCore.QRect(550, 440, 331, 231))
        self.img4.setObjectName("img4")
        self.btn_stop_1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop_1.setGeometry(QtCore.QRect(270, 80, 89, 25))
        self.btn_stop_1.setObjectName("btn_stop_1")
        self.btn_stop_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop_2.setGeometry(QtCore.QRect(840, 80, 89, 25))
        self.btn_stop_2.setObjectName("btn_stop_2")
        self.btn_stop_3 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop_3.setGeometry(QtCore.QRect(280, 380, 89, 25))
        self.btn_stop_3.setObjectName("btn_stop_3")
        self.btn_stop_4 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop_4.setGeometry(QtCore.QRect(840, 380, 89, 25))
        self.btn_stop_4.setObjectName("btn_stop_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 943, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.link_rstp.setText(_translate("MainWindow", "RTSP"))
        self.btn_add_cam.setText(_translate("MainWindow", "ADD CAM"))
        self.btn_delete_cam.setText(_translate("MainWindow", "DELETE CAM"))
        self.list_cam1.setItemText(0, _translate("MainWindow",
                                                 "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"))
        self.list_cam1.setItemText(1, _translate("MainWindow", "test.mp4"))
        self.btn_start_1.setText(_translate("MainWindow", "START1"))
        self.btn_start_2.setText(_translate("MainWindow", "START2"))
        self.list_cam2.setItemText(0, _translate("MainWindow",
                                                 "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"))
        self.list_cam2.setItemText(1, _translate("MainWindow", "test.mp4"))
        # self.img1.setText(_translate("MainWindow", "TextLabel"))
        # self.img2.setText(_translate("MainWindow", "TextLabel"))
        self.btn_start_3.setText(_translate("MainWindow", "START3"))
        self.list_cam3.setItemText(0, _translate("MainWindow",
                                                 "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"))
        self.list_cam3.setItemText(1, _translate("MainWindow", "test.mp4"))
        self.list_cam4.setItemText(0, _translate("MainWindow",
                                                 "rtsp://admin:atin%402022@192.168.1.232/profile1/media.smp"))
        self.list_cam4.setItemText(1, _translate("MainWindow", "test.mp4"))
        self.btn_start_4.setText(_translate("MainWindow", "START4"))
        # self.img3.setText(_translate("MainWindow", "TextLabel"))
        # self.img4.setText(_translate("MainWindow", "TextLabel"))
        self.btn_stop_1.setText(_translate("MainWindow", "STOP1"))
        self.btn_stop_2.setText(_translate("MainWindow", "STOP2"))
        self.btn_stop_3.setText(_translate("MainWindow", "STOP3"))
        self.btn_stop_4.setText(_translate("MainWindow", "STOP4"))


if __name__ == "__main__":
    import sys

    # Init QT
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())