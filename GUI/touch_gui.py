# gui/touch_gui.py

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2


class TouchGUI(QWidget):
    # Custom signal to pass video source back to main.py
    video_source_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Good Touch vs Bad Touch Detection")
        self.setGeometry(100, 100, 800, 600)
        self.cap = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.info_label = QLabel("Choose Input Mode")
        layout.addWidget(self.info_label)

        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.start_webcam)
        layout.addWidget(self.webcam_btn)

        self.video_btn = QPushButton("Select Video File")
        self.video_btn.clicked.connect(self.select_video)
        layout.addWidget(self.video_btn)

        self.setLayout(layout)

    def start_webcam(self):
        self.video_source_selected.emit(0)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                   "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_source_selected.emit(file_path)

    def display_frame(self, img):
        """Display a frame on the GUI."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()
