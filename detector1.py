# Common
import argparse
import os
import random
import sys
import time
from pathlib import Path
import numpy as np

# Torch
import torch
import torch.backends.cudnn as cudnn

# QT
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from PyQt5.QtCore import pyqtSignal

# Misc
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, plot_one_box
from utils.torch_utils import select_device, time_sync


# MAIN
class DetectorThread(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index=0):
        super().__init__()
        # id
        self.is_running = True
        self.index = index
        # VARIABLES
        self.cap = cv2.VideoCapture()
        self.source = 0  # file/dir/URL/glob, 0 for webcam
        self.model_file = 'yolov5n.pt'
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_crop = False  # save cropped prediction boxes
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.save_dir = None
        self.save_txt = False
        self.name = None


    def setup(self, a_source, a_model):
        self.source = a_source
        self.model_file = a_model

    @torch.no_grad()
    def run(self):

        print('Starting thread...', self.index)
        self.source = str(self.source)
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            self.source = check_file(self.source)  # download
        print(self.device)
        flag = torch.cuda.is_available()
        self.device = select_device(self.device)
        self.model = attempt_load(self.model_file, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        name_list = []
        self.cap.open(self.source)
        while self.is_running:
            flag, img = self.cap.read()
            # while self.is_running:
            prev_time = time.time()
            if img is not None:
                showimg = img
                with torch.no_grad():
                    img = letterbox(img, new_shape=self.imgsz)[0]
                    # Convert
                    # BGR to RGB, to 3x416x416
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0

                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # Inference
                    pred = self.model(img, augment=False)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                               max_det=self.max_det)
                    # Process detections

                    for i, det in enumerate(pred):  # detections per image
                        if det is not None and len(det):

                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], showimg.shape).round()
                            # Write results

                            for *xyxy, conf, cls in reversed(det):
                                curr_time = time.time()
                                fps = 1 / (curr_time - prev_time)
                                print(fps)
                                cv2.putText(showimg, f'FPS:{int(fps)}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0),
                                            3,
                                            cv2.LINE_AA)
                                label = '%s %.2f' % (self.names[int(cls)], conf)
                                name_list.append(self.names[int(cls)])
                                # print(label)
                                plot_one_box(
                                    xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

                # self.out.write(showimg)
                self.signal.emit(showimg)

                cv2.waitKey(1)
                # show = cv2.resize(showimg, (640, 640))
                # self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                # showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                #                          QtGui.QImage.Format_RGB888)
                # self.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
            else:
                self.cap.release()
                cv2.destroyAllWindows()
    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running = False

