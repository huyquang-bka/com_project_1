import pickle
import cv2
import numpy as np
import time
import socket

host = "192.168.1.33"
port = "8080"
count = 0
while count <= 10:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, int(port)))
    image = cv2.imread('download.jpg')
    image = cv2.resize(image, (100, 100))
    data = pickle.dumps(image)
    start = time.time()
    s.sendall(data)
    print(f"{count}: {time.time() - start}")
    s.close()

