import cv2
import base64
import time
import requests
import json

content_type = 'image/jpeg'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

image = cv2.imread(r"C:\Users\Admin\Downloads\c9_original.jpg")
byte_array = cv2.imencode('.jpg', image)[1].tobytes()
byte_base64 = base64.b64encode(byte_array).decode()
json_data = {'image_1': byte_base64, 'image_2': byte_base64}

# json_data = {"image": base64_bytes.decode('utf-8')}
r = requests.post("http://192.168.1.33:8000/image_post", json=json.dumps(json_data), headers=headers)
