from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


@app.route('/image', methods=['POST'])
def post_image():
    data = request.json
    # recv_dict = pickle.loads(data)
    for k, v in data.items():
        print(k, np.array(v, dtype=np.uint8).shape)

    # image = data['image']
    # image = np.array(image, dtype=np.uint8)
    # print(image.shape)
    return "Count objects: {}".format(len(data))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)
