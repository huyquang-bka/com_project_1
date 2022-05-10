from flask import Flask, request, jsonify
import numpy as np
import time
app = Flask(__name__)


@app.route('/image', methods=['POST'])
def post_image():
    data = request.json
    # recv_dict = pickle.loads(data)
    new_dict = {}
    for k, v in data.items():
        new_dict[k] = "Em gai"
    time.sleep(3)

    # image = data['image']
    # image = np.array(image, dtype=np.uint8)
    # print(image.shape)
    return new_dict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)
