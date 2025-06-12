import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class BasicCV(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Webcam")
        self.layout = QVBoxLayout()
        self.label = QLabel("Loading...")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication([])
    win = BasicCV()
    win.show()
    app.exec_()
