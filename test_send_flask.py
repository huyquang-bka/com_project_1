import requests
import cv2
import numpy as np
import time
import pickle

api = 'http://192.168.1.33:8000/image'
count = 0
while count <= 10:
    dict_send = dict()
    image = cv2.imread('download.jpg')
    H, W = image.shape[:2]
    for i in range(5):
        crop = image[i * H // 10:(i + 1) * H // 10, 0:W]
        dict_send[i] = np.array(crop, dtype=np.uint8).tolist()
    # frame_bytes = np.array(image, dtype=np.uint8).tolist()
    # data_send = pickle.dumps(dict_send)
    s = time.time()
    response = requests.post(api, json=dict_send)
    print(response.status_code, time.time() - s)
    count += 1
