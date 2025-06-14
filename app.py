import sys

from PyQt5.QtWidgets import QApplication
from flask import Flask, render_template, Response
import cv2
import threading
import time

from main import TouchDetectionApp

app = Flask(__name__)


class WebStream:
    def __init__(self):
        self.frame = None
        self.running = True

    def start(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, self.frame = cap.read()
            time.sleep(0.03)  # ~30 FPS
        cap.release()


stream = WebStream()


def generate_frames():
    while True:
        if stream.frame is not None:
            ret, buffer = cv2.imencode('.jpg', stream.frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Start camera thread
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = TouchDetectionApp()
    window.show()
    sys.exit(app.exec_())

    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)