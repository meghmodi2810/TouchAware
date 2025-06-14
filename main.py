import json
import sys
import threading
from datetime import datetime

import cv2
import pygame
import numpy as np
import winsound
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from joblib import dump, load

from Modules.HandDetectionModule import HandDetectionModule
from Modules.PoseDetectionModule import PoseDetectionModule
from Modules.InteractionAnalyzer import InteractionAnalyzer
from LearningModules.MLInteractionAnalyzer import MLInteractionAnalyzer

class TouchDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.training_sample_limit = 200
        self.init_audio_system()
        self.ml_analyzer = MLInteractionAnalyzer()
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

        self.training_mode = False
        self.training_data = []
        self.training_labels = []

    def toggle_training_mode(self):
        """Switch between detection and training modes"""
        self.training_mode = not self.training_mode
        self.status_label.setText("TRAINING MODE" if self.training_mode else "DETECTION MODE")

    def save_training_sample(self, label):
        """Save current frame's features with manual label"""
        if not hasattr(self, 'current_features'):
            print("[DEBUG] No features to save.")
            return

        if len(self.training_data) >= self.training_sample_limit:
            print("[INFO] Training data limit reached.")
            self.status_label.setText("LIMIT REACHED: Cannot save more")
            return

        self.training_data.append(self.current_features)
        self.training_labels.append(label)
        print(f"Saved sample #{len(self.training_data)} with label: {label}")
        self.status_label.setText(f"Samples Collected: {len(self.training_data)}")

    def train_model(self):
        """Train the ML model with collected data"""
        if len(self.training_data) == 0:
            return

        X = np.vstack(self.training_data)
        y = np.array(self.training_labels)

        # Train and save model
        self.ml_analyzer.model.fit(X, y)
        dump(self.ml_analyzer.model, 'touch_model.joblib')
        print("Model trained and saved!")




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
        ml_layout = QHBoxLayout()

        self.train_btn = QPushButton("Enable Training")
        self.train_btn.clicked.connect(self.toggle_training_mode)
        ml_layout.addWidget(self.train_btn)

        self.save_safe_btn = QPushButton("Save Safe Sample")
        self.save_safe_btn.clicked.connect(lambda: self.save_training_sample(0))
        ml_layout.addWidget(self.save_safe_btn)

        self.save_unsafe_btn = QPushButton("Save Unsafe Sample")
        self.save_unsafe_btn.clicked.connect(lambda: self.save_training_sample(2))
        ml_layout.addWidget(self.save_unsafe_btn)

        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        ml_layout.addWidget(self.train_model_btn)

        layout.addLayout(ml_layout)

        self.status_label = QLabel("DETECTION MODE")
        layout.addWidget(self.status_label)

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

            self.body_points = self._get_body_points(pose_landmarks, img)
            img = self.apply_privacy_protection(img)
            self.draw_safety_zones(img, self.body_points)
            self.display_image(img)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
        finally:
            self.is_processing = False

        if self.training_mode and hands_landmarks and body_points:
            # Extract and store features
            self.current_features = self.ml_analyzer.extract_features(
                [(x, y) for _, _, x, y in hands_landmarks[0]],
                body_points
            )

            # Visual feedback
            cv2.putText(img, "TRAINING MODE - Label Current Pose",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)



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
                    risk_level = 2 if zone in ['groin', 'buttocks'] else 1
                    self.play_alert(risk_level)
                    self.log_incident(zone, risk_level)

                    if current_time - self.last_alert_time > 2000:
                        self.alert_sound.play()
                        self.play_alert(risk_level)
                        self.log_incident(zone, risk_level)
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

    def init_audio_system(self):
        """Robust audio initialization with fallbacks"""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            self.alert_sound = pygame.mixer.Sound("Sounds/alert.wav")
            self.warning_sound = pygame.mixer.Sound("Sounds/warning.wav")
            self.alert_channel = pygame.mixer.Channel(0)  # Dedicated channel
        except Exception as e:
            print(f"Audio init failed: {e}. Using system sounds.")
            self.use_system_sounds = True

    def play_alert(self, level):
        """Level: 1 (warning), 2 (emergency)"""
        try:
            if hasattr(self, 'use_system_sounds'):
                threading.Thread(target=lambda: winsound.Beep(800 if level == 1 else 2000, 1000), daemon=True).start()
            else:
                sound = self.warning_sound if level == 1 else self.alert_sound
                if not self.alert_channel.get_busy():
                    self.alert_channel.play(sound)
        except Exception as e:
            print(f"Couldn't play sound: {e}")

    def apply_privacy_protection(self, img):
        """Blur sensitive areas in saved recordings"""
        if not hasattr(self, 'body_points'):
            return img

        # Blur groin/buttocks areas
        for zone in ['groin', 'buttocks']:
            if zone in self.body_points:
                x, y = self.body_points[zone]
                img[y - 50:y + 50, x - 30:x + 30] = cv2.GaussianBlur(
                    img[y - 50:y + 50, x - 30:x + 30], (51, 51), 0
                )
        return img

    def init_emergency_contacts(self):
        """Load trusted contacts from config"""
        try:
            with open('config/contacts.json') as f:
                self.emergency_contacts = json.load(f)
        except:
            self.emergency_contacts = ["guardian@example.com", "+1234567890"]

    def send_emergency_alert(self, zone):
        """Send SMS/email on high-risk incidents"""
        if zone in ['groin', 'buttocks']:
            message = f"EMERGENCY: Inappropriate contact detected ({zone}) at {datetime.now()}"
            # Implement your preferred notification method:
            # - SMTP for emails
            # - Twilio API for SMS
            print(f"ALERT SENT: {message}")

    def draw_safety_zones(self, img, body_points):
        """Color-coded zone visualization"""
        zone_colors = {
            'groin': (0, 0, 255),  # Red
            'buttocks': (0, 0, 200),  # Dark red
            'chest': (0, 165, 255),  # Orange
            'mouth': (0, 255, 255)  # Yellow
        }

        for zone, point in body_points.items():
            if zone in zone_colors:
                cv2.circle(img, point, 25, zone_colors[zone], 2)
                cv2.putText(img, zone.upper(),
                            (point[0] - 30, point[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            zone_colors[zone], 1)


    def log_incident(self, zone, risk_level):
        def _log():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open("config/incident_log.txt", "a") as f:
                f.write(f"{timestamp}, Zone={zone}, Risk Level={risk_level}\n")

        threading.Thread(target=_log, daemon=True).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TouchDetectionApp()
    window.show()
    sys.exit(app.exec_())