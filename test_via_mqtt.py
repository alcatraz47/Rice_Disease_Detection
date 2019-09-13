import paho.mqtt.client as mqtt
import time
import json
import base64
import cv2
from PIL import Image


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

def on_message(client,userdata,msg):
    bytes_value = msg.payload.decode('utf8')
    print(bytes_value)
    # img = stringToRGB(bytes_value)
    try:
        # img = bytes_value.decode('base64')
        img = base64.b64decode(str(bytes_value))
        topic_publish = "499-response"
        file_name = 'img.png'
        with open(file_name, 'wb') as file:
            file.write(img)
        file.close()
        image = cv2.imread('img.png', -1)
        prediction = test_model(image)
        if prediction == 0:
            client.publish("499-response", payload = "4")
        elif prediction == 1:
            client.publish("499-response", payload = "3")
        elif prediction == 2:
            client.publish("499-response", payload = "0")
        elif prediction == 3:
            client.publish("499-response", payload = "5")
        elif prediction == 4:
            client.publish("499-response", payload = "1")
        elif prediction == 5:
            client.publish("499-response", payload = "2")


        # client.subscribe(topic_publish)
        print("test_accessed")
    except Exception as e:
        print(e)

def test_model(img):
    # load json and create model
    json_file = open('cnn_model_83.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("cnn_model_83.h5")
    print("Loaded model from disk")

    CLASSES = ['BacterialLeafBlight', 'BrownSpot', 'Healthy', 'Hispa', 'LeafBlast', 'LeafSmut']
    img = np.array(img).reshape(-1, 120, 120, 3)
    prediction = loaded_model.predict_classes(img)
    return prediction

client = mqtt.Client()
client.connect("3.0.109.66")

topic_response = "499"
client.subscribe(topic_response) 
client.on_message = on_message


# # client = mqtt.Client()
# # client.connect("13.58.86.17")
# topic_publish = "499-response"
# # client.subscribe(topic_publish)
# client.publish("499-response", payload = "1")
client.loop_forever()