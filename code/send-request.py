import base64
from matplotlib import pyplot as plt
import cv2
import io
import tensorflow as tf
import numpy as np
import base64
import json
import requests

def preprocess(path):
    img = cv2.imread(path)
    resize = tf.image.resize(img, (256, 256))
    ret = np.expand_dims(resize/255, 0)
    return ret

path = "new_test_data/not_hot_dog/4.jpg"
ppimg = preprocess(path)
data = json.dumps({
    "instances": ppimg.tolist()
})
headers = {"content-type": "application/json"}

response = requests.post('http://localhost:8501/v1/models/seefood-model:predict', data=data, headers=headers)
result = response.json()['predictions'][0][0]
print(result)
