
import cv2
import keras;
from flask import Flask, render_template, request, Response

app=Flask(__name__)
model = keras.models.load_model(
            'C:\\Users\\User\\OneDrive\\Documents\\FaceRecognitionFlask\\facerecognition.h5')
model1 = keras.models.load_model(
            'C:\\Users\\User\\OneDrive\\Documents\\FaceRecognitionFlask\\mask_detection.h5')
camera=cv2.VideoCapture(0)
output=[]#("message stark","hi")]
@app.route('/')
def home_page():
    return render_template("IY_Home_page.html",result=output)
@app.route('/about')
def about_page():
    return render_template("about.html",result=output)
@app.route('/contact')
def contact_page():
    return render_template("contact.html",result=output)
@app.route('/cam')
def cam():
    return render_template("Webcam.html",result=output)



def gen():
    while True:
        ret,frame=camera.read();
        image = cv2.resize(frame, (224, 224))
        li = ['chaya', 'manoj', 'queany', 'rakesh', 'sambhrama', 'sankalpa', 'sapana', 'shradda', 'shreya', 'sowmya',
              'suprabha', 'swathi', 'thrupthi', 'vaishali', 'vaishu']
        haar = cv2.CascadeClassifier('C:\\haarcascades\\haarcascade_eye.xml')
        haar1 = cv2.CascadeClassifier('C:\\haarcascades\\haarcascade_frontalface_default.xml')
        eyes = haar.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        face = haar1.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

        y_pred1=model1.predict_classes(image.reshape(1, 224, 224, 3))
        y_pred = int(model.predict_classes(image.reshape(1, 224, 224, 3)))
        if y_pred1 == 0:
            name = "No Mask " + li[y_pred]
        else:
            name = "With Mask " + li[y_pred]
        for x, y, w, h in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for x, y, w, h in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame=jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result',methods=["POST","GET"])
def Result():
        print(output)
        return render_template("IY_Home_page.html",result=output)

if __name__=="__main__":
    app.run(debug=True)#,host="192.168.43.161")



