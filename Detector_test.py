# Common
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import time

# Torch
import torch
import torch.backends.cudnn as cudnn

# QT
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from PyQt5.QtCore import pyqtSignal, QObject
from Wang.wang_funtion import post_data

# Misc
from models.common import DetectMultiBackend, letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# MAIN
class DetectorThread(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index=0, parent=None):
        super().__init__()
        # super(DetectorThread, self).__init__(parent)
        # id
        self.index = index
        self.is_running = True
        # VARIABLES
        # file/dir/URL/glob, 0 for webcam
        self.source = r"C:\Users\Admin\Downloads\video-1636524259.mp4"
        self.weights = 'yolov5s.pt'
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_crop = False  # save cropped prediction boxes
        # filter by class: --class 0, or --class 0 2 3
        self.classes = range(80)
        self.agnostic_nms = False  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = True  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def setup(self, a_source, a_model):
        self.source = int(a_source) if a_source == "0" else a_source
        self.model_file = a_model

    @torch.no_grad()
    def run(self):
        count = 0
        cap = cv2.VideoCapture(self.source)
        while self.is_running:
            count += 1
            s = time.time()
            ret, im0 = cap.read()
            if not ret:
                break
            # img0 = cv2.resize(im0, (100, 100))
            # status_code = post_data("http://192.168.1.31:5000/image", img0, count)
            status_code = 1
            print(f"{self.index} {status_code} {time.time() - s} ")
            self.signal.emit(im0)
            cv2.waitKey(1)

    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running = False
