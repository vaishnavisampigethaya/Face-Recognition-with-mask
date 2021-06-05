
import cv2
import pickle

import keras
from imutils.video import WebcamVideoStream
import tensorflow

class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        image = self.stream.read()
        image = cv2.resize(image, (224, 224))
        li = ['chaya', 'manja', 'queany', 'rakesh', 'sam', 'sankalpa', 'sapana', 'shradda', 'shreya', 'sowmya', 'suprabha', 'swathi', 'thrupthi', 'vaishali', 'vaishu']
        haar = cv2.CascadeClassifier('haarcascade_eye.xml')
        haar1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eyes = haar.detectMultiScale(image)
        face = haar1.detectMultiScale(image)
        model=keras.models.load_model('facerecognition.h5')
        y_pred = int(model.predict_classes(image.reshape(1, 224, 224, 3)))
        #y_pred1 = model1.predict_classes(image.reshape(1, 224, 224, 3))
        name = li[y_pred]
        for  (top, right, bottom, left) in eyes:
            startX = int(left)
            startY = int(top)
            endX = int(right)
            endY = int(bottom)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        for  (top, right, bottom, left) in face:
            startX = int(left)
            startY = int(top)
            endX = int(right)
            endY = int(bottom)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            cv2.putText(image, name, (endX - 70, endY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        data = []
        data.append(image.tobytes())
        data.append(name)
        return data