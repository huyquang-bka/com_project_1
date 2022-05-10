# Common
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import opt
from utils.general import xyxy2xywh
import requests

id_dict = {}
res_dict = {}
mutex = QtCore.QMutex()


class PostApi(QtCore.QThread):
    signal = pyqtSignal(dict)

    def __init__(self, index=0, parent=None):
        super().__init__()
        self.index = index
        self.api = "http://192.168.1.33:8000/image"
        self.is_running = True

    def run(self):
        global res_dict
        while self.is_running:
            s = time.time()
            send_dict = {}
            mutex.lock()
            for key in id_dict.keys():
                send_dict[key] = id_dict[key]
            res_dict = requests.post(self.api, json=send_dict).text
            res_dict = json.loads(res_dict)
            # res_dict = {"1": "Em gai"}
            print("Class PostApi: ", res_dict)
            # self.signal.emit(res_dict)
            mutex.unlock()
            time.sleep(0.0001)

    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running = False


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
        self.classes = [0]  # (80) range of classes
        self.agnostic_nms = True  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = True  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def setup(self, a_source, a_model):
        self.source = a_source
        self.weights = a_model

    @torch.no_grad()
    def run(self):
        # Load deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
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
        global id_dict
        global res_dict
        while self.is_running:
            ret, img0 = cap.read()
            img0 = cv2.resize(img0, (1280, 720))
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
            # Process predictions
            id_dict_local = {}
            spot_dict_local = {}
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    xyxys, confs, clss = [], [], []
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # cv2.imwrite(f"LP_image/{x1}_{y1}_{x2}_{y2}.jpg", im0[y1:y2, x1:x2])
                        xyxys.append([x1, y1, x2, y2])
                        confs.append(conf)
                        clss.append(cls)

                    xywhs = xyxy2xywh(torch.Tensor(xyxys))
                    confs = torch.Tensor(confs)
                    clss = torch.tensor(clss)
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            x1, y1, x2, y2 = output[0:4]
                            id = output[4]
                            crop = im0[y1:y2, x1:x2]
                            # if count % 5 == 0:
                            id_dict_local[str(id)] = np.array(crop, dtype=np.uint8).tolist()
                            spot_dict_local[str(id)] = [x1, y1, x2, y2]
                            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # if count % 5 == 0:
                mutex.lock()
                del id_dict
                id_dict = {}
                for i in range(5):
                    id_dict[str(i)] = 1
                for key in res_dict:
                    if key in spot_dict_local.keys():
                        x1, y1, x2, y2 = spot_dict_local[key]
                        name = res_dict[key]
                        cv2.putText(im0, f"{name}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("Class Images:", res_dict, spot_dict_local)
                mutex.unlock()
            FPS = 1 // (time.time() - s)

            cv2.putText(im0, '%g FPS' % FPS, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Thread {self.index} FPS: {FPS}")
            self.signal.emit(im0)
            cv2.waitKey(1)

    def stop(self):
        print('Stopping thread...', self.index)
        self.is_running = False
