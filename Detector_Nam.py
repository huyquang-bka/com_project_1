import os
import random
from queue import Queue
# Torch
import requests
# QT
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
# Misc
# limit the number of cpus used by high performance libraries
import os
from Yolov5_DeepSort_OSNet.yolov5.utils.augmentations import letterbox

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')

import os

import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from Yolov5_DeepSort_OSNet.yolov5.models.experimental import attempt_load
from Yolov5_DeepSort_OSNet.yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS, IMG_FORMATS
from Yolov5_DeepSort_OSNet.yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from Yolov5_DeepSort_OSNet.yolov5.utils.torch_utils import select_device, time_sync
from Yolov5_DeepSort_OSNet.yolov5.utils.plots import Annotator, colors, save_one_box, plot_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# MAIN
class DetectorThread(QtCore.QThread):
    signal = pyqtSignal(np.ndarray)
    signal1 = pyqtSignal(str)

    def __init__(self, index=0):

        super().__init__()
        # id
        self.is_running = True
        self.index = index
        # VARIABLES
        self.cap = cv2.VideoCapture()
        self.deep_sort_model = 'osnet_x0_25'
        self.config_deepsort = '/home/namnd/PycharmProjects/Yolov5_DeepSort_OSNet/deep_sort/configs/deep_sort.yaml'
        self.source = 0  # file/dir/URL/glob, 0 for webcam
        self.model_file = 'yolov5n.pt'
        self.data = '/home/namnd/PycharmProjects/yolov5/data/coco128.yaml'  # dataset.yaml path
        self.imgsz = (640, 480)  # inference size (height, width)
        self.conf_thres = 0.5  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 20  # maximum detections per image
        self.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_crop = False  # save cropped prediction boxes
        self.classes = 0  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.line_thickness = 1  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.save_dir = None
        self.save_txt = False
        self.name = None
        self.send_dict = {}
        self.queue = Queue()

    def setup(self, a_source, a_model):
        self.source = a_source
        self.model_file = a_model

    # @app.post("/image/")
    @torch.no_grad()
    def run(self):

        print('Starting thread...', self.index)
        self.source = str(self.source)
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            self.source = check_file(self.source)  # download
        self.device = select_device(self.device)
        self.model = attempt_load(
            self.model_file, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        webcam = self.source == '0' or self.source.startswith(
            'rtsp') or self.source.startswith('http') or self.source.endswith('.txt')
        if webcam:
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=True)
            nr_sources = len(dataset)
        else:
            nr_sources = 1
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)

        # Create as many trackers as there are video sources
        deepsort_list = []
        for i in range(nr_sources):
            deepsort_list.append(
                DeepSort(
                    self.deep_sort_model,
                    self.device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                )
            )
        outputs = [None] * nr_sources

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        name_list = []
        self.cap.open(self.source)

        while self.is_running:
            flag, img = self.cap.read()
            area1 = [(600, 72), (650, 419), (802, 419), (802, 72)]
            area2 = [(930, 206), (1120, 1060), (1460, 1060), (1010, 56)]
            if not flag:
                break
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
                    for area in [area1, area2]:
                        cv2.polylines(showimg, [np.array(area, np.int32)], True, (15, 220, 10), 6)
                        # cv2.polylines(showimg, [np.array(area2, np.int32)], True, (15, 220, 10), 6)
                    for i, det in enumerate(pred):  # detections per image
                        if det is not None and len(det):

                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                            # Print results
                            s = str('')
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            xywhs = xyxy2xywh(det[:, 0:4])
                            confs = det[:, 4]
                            clss = det[:, 5]

                            # pass detections to deepsort
                            t4 = time_sync()
                            outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), showimg)
                            t5 = time_sync()
                            # fps = int(1 / (t5 - t4))
                            # draw boxes for visualization

                            if len(outputs[i]) > 0:
                                for j, (output) in enumerate(outputs[i]):
                                    bboxes = output[0:4]
                                    x1, y1, x2, y2 = int(output[0]), int(output[1]), int(output[2]), int(output[3])
                                    cx = int((x1 + x2) / 2)
                                    cy = int((y1 + y2) / 2)
                                    point = (x1, y1)
                                    point1 = (x2, y2)
                                    point2 = (x1, y2)
                                    point3 = (x2, y1)
                                    result_1 = []
                                    result_2 = []
                                    for point_ in [point, point1, point2, point3]:
                                        result1 = cv2.pointPolygonTest((np.array(area2, np.int32)), point_, False)
                                        result2 = cv2.pointPolygonTest((np.array(area1, np.int32)), point_, False)
                                        result_1.append(result1)
                                        result_2.append(result2)

                                    id = output[4]
                                    cls = output[5]
                                    conf = output[6]

                                    for r1, r2 in zip(result_1, result_2):
                                        if r1 >= 0 or r2 >= 0:
                                            c = int(cls)  # integer class
                                            label = f'{id:0.0f} {self.names[c]} {conf:.2f}'
                                            plot_one_box(
                                                bboxes, showimg, label=label, color=self.colors[int(cls)],
                                                line_thickness=2)
                                        # annotator.box_label(bboxes, label, color=colors(c, True))

                                    fps = int(1 / (time.time() - prev_time))
                                    cv2.putText(showimg, f'FPS:{int(fps)}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (100, 255, 0),
                                                3,
                                                cv2.LINE_AA)

                                    # crop = showimg[y1:y2, x1:x2]
                                    # t2 = Thread(target=self.crop_image, args=(showimg, x1, x2, y1, y2, self.queue), daemon=True)
                                    # print(t2)

                                    # self.send_dict[str(id)] = np.array(crop, dtype=np.uint8).tolist()
                                    # t3 = Thread(target=self.image2array, args=(id, self.queue.get()))
                                    # t2.start()
                                    # t3.start()
                                    # t2.join()
                                    # t3.join()

                    # try:
                    #     s2 = time.time()
                    #     response = requests.post('http://127.0.0.1:5000/image', json=self.send_dict, timeout=0.5).text
                    #     # t1 = Thread(target=self.send_data, args=(self.send_dict, self.queue))
                    #     # t1.start()
                    #     # print(type(response))
                    #     # self.signal1.emit(response)
                    #     print(response + " time: " + str(time.time() - s2) + f" {self.index}")
                    #     # print(self.queue.get()+" time: " + str(time.time() - s2) + f" {self.index}")
                    # except:
                    #     print("error post")
                    # print(f"Thread {self.index} FPS: {fps}")

                # self.out.write(showimg)
                #                 img = decode(showimg)
                #                 print(img)
                self.signal.emit(showimg)

                cv2.waitKey(1)
            else:
                self.cap.release()
            # return name_list

    # @app.get("/")
    # async def home(self):
    #     return "welcome to my website"

    def stop(self):
        print('Stopping thread...', self.index)

        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def send_data(self, data, queue):
        response = requests.post('http://127.0.0.1:5000/image', json=data).text
        return queue.put(response)

    def crop_image(self, image, x1, x2, y1, y2, queue):
        crop = image[y1:y2, x1:x2]
        crop = queue.put(crop)
        return crop

    def image2array(self, id, crop):
        self.send_dict[str(id)] = np.array(crop, dtype=np.uint8).tolist()
