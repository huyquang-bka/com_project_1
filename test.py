import cv2
import numpy as np
import mediapipe as mp
import time
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import opt
from utils.general import xyxy2xywh
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cfg = get_config()
cfg.merge_from_file(opt.config_deepsort)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

cap = cv2.VideoCapture("2.mp4")
while True:
    ret, image = cap.read()
    if not ret:
        break
    print("Camera is working...")
    s = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    xyxys, confs, clss = [], [], []
    if results.detections:
        for detection in results.detections:
            relative_bb = detection.location_data.relative_bounding_box
            score = detection.score
            id = detection.detection_id
            print(id)
            x = int(relative_bb.xmin * image.shape[1])
            y = int(relative_bb.ymin * image.shape[0])
            w = int(relative_bb.width * image.shape[1])
            h = int(relative_bb.height * image.shape[0])
            x2 = x + w
            y2 = y + h

            xyxys.append([x, y, x2, y2])
            confs.append(score)
            clss.append(0)
        xywhs = xyxy2xywh(torch.Tensor(xyxys))
        confs = torch.Tensor(confs)
        clss = torch.tensor(clss)
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, image)

        id_dict_local = {}
        spot_dict_local = {}
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                x1, y1, x2, y2 = output[0:4]
                id = output[4]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    fps = 1 // (time.time() - s)
    cv2.putText(image, "FPS: {:.1f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Face Detection", image)
    key = cv2.waitKey(1)
    if key == 27:
        break
