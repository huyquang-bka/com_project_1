from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(407, 474)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_start_cam2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_cam2.setObjectName("btn_start_cam2")
        self.gridLayout.addWidget(self.btn_start_cam2, 2, 1, 1, 1)
        self.btn_start_cam1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start_cam1.setObjectName("btn_start_cam1")
        self.gridLayout.addWidget(self.btn_start_cam1, 0, 1, 1, 1)
        self.btn_stop_cam1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop_cam1.setObjectName("btn_stop_cam1")
        self.gridLayout.addWidget(self.btn_stop_cam1, 0, 2, 1, 1)
        self.btn_stop_cam2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop_cam2.setObjectName("btn_stop_cam2")
        self.gridLayout.addWidget(self.btn_stop_cam2, 2, 2, 1, 1)
        self.img1 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img1.sizePolicy().hasHeightForWidth())
        self.img1.setSizePolicy(sizePolicy)
        self.img1.setStyleSheet("background:white;")
        self.img1.setText("")
        self.img1.setObjectName("img1")
        self.gridLayout.addWidget(self.img1, 1, 1, 1, 2)
        self.img2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img2.sizePolicy().hasHeightForWidth())
        self.img2.setSizePolicy(sizePolicy)
        self.img2.setStyleSheet("background:white;")
        self.img2.setText("")
        self.img2.setObjectName("img2")
        self.gridLayout.addWidget(self.img2, 3, 1, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_start_cam2.setText(_translate("MainWindow", "Start Camera 2"))
        self.btn_start_cam1.setText(_translate("MainWindow", "Start Camera 1"))
        self.btn_stop_cam1.setText(_translate("MainWindow", "Stop Camera 1"))
        self.btn_stop_cam2.setText(_translate("MainWindow", "Stop Camera 2"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
