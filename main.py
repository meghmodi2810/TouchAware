import sys
import cv2
import pygame
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from Modules.HandDetectionModule import HandDetectionModule
from Modules.PoseDetectionModule import PoseDetectionModule
from Modules.InteractionAnalyzer import InteractionAnalyzer


class TouchDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Good Touch vs Bad Touch Detection")
        self.setGeometry(100, 100, 800, 600)

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.alert_sound = pygame.mixer.Sound("Sounds/alert.wav")
        self.last_alert_time = 0

        # Initialize detection modules
        self.hand_detector = HandDetectionModule(maxHands=2)
        self.pose_detector = PoseDetectionModule()
        self.analyzer = InteractionAnalyzer(safe_threshold=50)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_processing = False

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Choose Input Mode")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.use_webcam)
        layout.addWidget(self.webcam_btn)

        self.video_btn = QPushButton("Open Video File")
        self.video_btn.clicked.connect(self.open_video)
        layout.addWidget(self.video_btn)

        self.setLayout(layout)

    def use_webcam(self):
        self._start_capture(0)

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self._start_capture(file_path)

    def _start_capture(self, source):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.label.setText("Error: Could not open video source")
            return

        # Set input mode (webcam or video file)
        is_webcam = source == 0  # True if source is 0 (webcam), False otherwise
        self.hand_detector.set_input_mode(is_webcam)

        self.label.setText("Processing...")
        self.timer.start(30)

    def update_frame(self):
        if self.is_processing:
            return

        self.is_processing = True
        try:
            success, img = self.cap.read()
            if not success:
                self.timer.stop()
                self.label.setText("Video ended or error reading frame")
                return

            # Process hands
            img, hands_data = self.hand_detector.multiHandFinder(img, draw=True)
            hands_landmarks = self.hand_detector.multiHandPositionFinder(img, draw=True)

            # Process pose
            img = self.pose_detector.poseFinder(img, draw=True)
            pose_landmarks = self.pose_detector.positionFinder(img, draw=True)

            # Extract body points
            body_points = self._get_body_points(pose_landmarks, img)

            # Analyze interactions
            if hands_landmarks and body_points:
                self._analyze_interactions(img, hands_landmarks, body_points)

            self.display_image(img)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
        finally:
            self.is_processing = False

    def _get_body_points(self, pose_landmarks, img):
        body_points = {}
        if not pose_landmarks or len(pose_landmarks) < 25:
            return body_points

        try:
            # Get shoulders
            l_shoulder = pose_landmarks[11][1:]
            r_shoulder = pose_landmarks[12][1:]
            chest = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
            cv2.circle(img, chest, 10, (0, 0, 255), cv2.FILLED)
            body_points['chest'] = chest

            # Get hips
            l_hip = pose_landmarks[23][1:]
            r_hip = pose_landmarks[24][1:]
            waist = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
            cv2.circle(img, waist, 10, (255, 0, 0), cv2.FILLED)
            body_points['waist'] = waist

            # Add more body points as needed
        except IndexError:
            pass

        return body_points

    def _analyze_interactions(self, img, hands_landmarks, body_points):
        for hand in hands_landmarks:
            if not hand:
                continue

            hand_label = hand[0][0] if hand else 'unknown'
            hand_fingertips = [(x, y) for _, id, x, y in hand if id in [8, 12, 16, 20]]  # Include more fingertips

            if hand_fingertips:
                zone, distance = self.analyzer.analyze(hand_fingertips, body_points)
                if zone != "none":
                    current_time = pygame.time.get_ticks()

                    if current_time - self.last_alert_time > 100:
                        self.alert_sound.play()
                        self.last_alert_time = current_time

                    cv2.putText(
                        img, f"{hand_label.title()} hand near {zone.upper()}",
                        (10, 60 if hand_label == 'left' else 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

    def display_image(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"Error displaying image: {str(e)}")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TouchDetectionApp()
    window.show()
    sys.exit(app.exec_())