import cv2
from numpy import dtype
import requests
import numpy as np


def post_data(api, im_arr, frame_count):                   
    # headers = {'Content-type': 'image/jpeg'}
    frame_bytes = np.array(im_arr, dtype=np.uint8).tolist()
    payload = {"image": frame_bytes, "frame_count": str(frame_count)}
    response = requests.post(api, json=payload)
    return response.status_code
