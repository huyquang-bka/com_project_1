import numpy as np
import time

# QT
from PyQt5 import QtCore

from PyQt5.QtCore import pyqtSignal, QObject
import cv2
import requests
import json

# MAIN
id_dict = {}
res_dict = {}

mutex = QtCore.QMutex()
# is_running = True


class PostApi(QtCore.QThread):
    signal = pyqtSignal(dict)

    def __init__(self, index=0, parent=None):
        super().__init__()
        self.index = index
        self.api = "http://192.168.1.33:8000/image"
        self.is_running = True
        # self.mutex = QtCore.QMutex()
        # self.mutex = mutex

    def run(self):
        # global is_running
        # global mutex
        global res_dict
        while self.is_running:
            send_dict = {}
            mutex.lock()
            for key in id_dict.keys():
                send_dict[key] = id_dict[key]
            # res_dict = requests.post(self.api, json=send_dict).text
            # res_dict = json.loads(res_dict)
            print("Class PostApi: ", res_dict)
            mutex.unlock()
            self.signal.emit(send_dict)
            time.sleep(0.001)

    def stop(self):
        # global is_running
        print('Stopping thread...', self.index)
        self.is_running = False
        # is_running = False


class DetectorThread(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index=0, parent=None):
        super().__init__()
        # super(DetectorThread, self).__init__(parent)
        # id
        self.index = index
        self.is_running = True
        # self.mutex = QtCore.QMutex()
        # self.mutex = mutex
        # VARIABLES
        # file/dir/URL/glob, 0 for webcam
        self.source = r"C:\Users\Admin\Downloads\video-1636524259.mp4"

    def setup(self, a_source, a_model):
        self.source = int(a_source) if a_source == "0" else a_source

    def run(self):
        global id_dict
        # global mutex
        # global is_running
        count = 0
        cap = cv2.VideoCapture(self.source)
        while self.is_running:
            # is_running = True
            count += 1
            s = time.time()
            ret, im0 = cap.read()
            if not ret:
                # cap = cv2.VideoCapture(self.source)
                break
            mutex.lock()
            del id_dict
            id_dict = {}
            for i in range(5):
                id_dict[str(i)] = 1
            local_dict = id_dict.copy()
            for key in res_dict:
                try:
                    local_dict[key] = res_dict[key]
                except:
                    pass
            print("Class DetectorThread: ", local_dict)
            mutex.unlock()
            self.signal.emit(im0)
            # try:
            #     print(f"Index {self.index} fps:", 1 // (time.time() - s))
            # except:
            #     print(f"Index {self.index} fps > 1000")
            print(f"Fps thread {self.index}:", 1 / (time.time() - s))
            cv2.waitKey(1)

    def stop(self):
        # global is_running
        print('Stopping thread...', self.index)
        self.is_running = False
        # is_running = False
