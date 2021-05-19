import os
import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2

model1 = load_model('selfDriveModel.h5')

sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10

def preprocess(img):
    img = img[60:130,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img

@sio.on('connect')
def connect(sid,environ):
    print('Connected')
    sendControl(0,0)
    
def sendControl(steering,throttle):
    sio.emit('steer',data = {'steering_angle':steering.__str__(),
                            'throttle':throttle.__str__()
                            })

@sio.on('telemetry')
def telemetry(sid,data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocess(image)
    image = np.array([image])
    steering = float(model1.predict(image))
    throttle = 1.0 - speed / maxSpeed #steering**2 - (speed/maxSpeed)**2
    print('Steering = {:.2f}, Throttle = {:.2f}, Speed = {:.2f}'.format(steering,throttle,speed))
    sendControl(steering,throttle)
    
if __name__ == '__main__':
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)