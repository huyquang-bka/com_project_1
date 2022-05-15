from flask import Flask, request, jsonify
import numpy as np
import time
import base64
import cv2
import json

app = Flask(__name__)


@app.route('/image', methods=['POST'])
def post_image():
    data = request.json
    print(data.keys())
    # recv_dict = pickle.loads(data)
    new_dict = {}
    for k, v in data.items():
        new_dict[k] = "Em gai" + str(k)
        image = base64.b64encode(v.encode('utf-8'))
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        print(image.shape)
        cv2.imwrite("Received_data/{}.jpg".format(k), image)
    time.sleep(2)

    return new_dict


@app.route('/image_post', methods=['POST'])
def image_post():
    r = request.json
    r = json.loads(r)
    new_dict = {}
    for k, v in r.items():
        image = base64.b64decode(v.encode('utf-8'))
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("Received_data/{}.jpg".format(k), image)
        new_dict[k] = "Em gai " + str(k)
    time.sleep(2)
    return new_dict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)
