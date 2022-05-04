import cv2
import numpy as np
from Wang.wang_funtion import *
import time
import socket
import pickle

count = 0
host = "192.168.1.31"
port = 8080
while count < 10:
    img = cv2.imread('download.jpg')
    # img = cv2.resize(img, (100, 100))
    start = time.time()
    # status_code = post_data("http://192.168.1.31:5000/image", img, 1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(pickle.dumps(img))
    print(time.time() - start)
    count += 1
    s.close()
