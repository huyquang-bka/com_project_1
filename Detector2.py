# Common
from pathlib import Path
import numpy as np
import time
# Torch
import torch
import torch.backends.cudnn as cudnn

# QT
from PyQt5 import QtCore

# Misc
from models.common import DetectMultiBackend, letterbox
from utils.general import (check_img_size, cv2, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
import time
from Wang.wang_funtion import *


# MAIN
class DetectorThread(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index=0, parent=None):
        super().__init__()
        # super(DetectorThread, self).__init__(parent)
        # id
        self.is_running = True
        self.index = index
        # VARIABLES
        self.source = r"C:\Users\Admin\Downloads\video-1636524259.mp4"  # file/dir/URL/glob, 0 for webcam
        self.weights = 'yolov5s.pt'
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = 640  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_crop = False  # save cropped prediction boxes
        self.classes = range(80)  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def setup(self, a_source, a_model):
        self.source = a_source
        self.model_file = a_model

    @torch.no_grad()
    def run(self):
        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.model.half() if self.half else model.model.float()
        print('Starting thread...', self.index)
        self.source = str(self.source)
        cap = cv2.VideoCapture(self.source)
        count = 0
        while self.is_running:
            ret, img0 = cap.read()
            if not ret:
                break
            count += 1
            s = time.time()
            im0 = img0.copy()
            img = letterbox(img0, self.imgsz, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(img)
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # cv2.imwrite(f"LP_image/{x1}_{y1}_{x2}_{y2}.jpg", im0[y1:y2, x1:x2])
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 3)
            FPS = 1 // (time.time() - s)
            status_code = post_data("http://localhost:5000/image", im0, count)
            print(status_code)
            cv2.putText(im0, '%g FPS' % FPS, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.signal.emit(im0)
            cv2.waitKey(1)

    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running=False
