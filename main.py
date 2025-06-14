import os
import sys
import json
import threading
import time
from datetime import datetime
import cv2
import pygame
import numpy as np
import winsound
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox,
                             QComboBox, QSlider, QSpinBox, QProgressDialog)
from PyQt5.QtCore import QTimer, Qt, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from joblib import dump, load

from Modules.HandDetectionModule import HandDetectionModule
from Modules.PoseDetectionModule import PoseDetectionModule
from Modules.InteractionAnalyzer import InteractionAnalyzer
from LearningModules.MLInteractionAnalyzer import MLInteractionAnalyzer, TrainingThread


class TouchDetectionApp(QWidget):

    config_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        # Load configuration
        self.config = self.load_config()
        self.training_sample_limit = self.config.get('training_sample_limit', 200)

        # Initialize systems
        self.init_audio_system()
        self.init_ui()
        self.init_detection_modules()
        self.init_emergency_contacts()

        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.avg_fps = 0
        self.last_alert_time = 0

        # State variables
        self.is_processing = False
        self.training_mode = False
        self.training_data = []
        self.training_labels = []

        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(1000 // self.config.get('target_fps', 30))



    def toggle_training_mode(self):
        """Switch between detection and training modes"""
        self.training_mode = not self.training_mode
        mode_text = "Training" if self.training_mode else "Detection"
        self.mode_label.setText(f"Mode: {mode_text}")
        self.train_mode_btn.setText("Disable Training" if self.training_mode else "Enable Training")

        # Update button states
        self.save_safe_btn.setEnabled(self.training_mode)
        self.save_unsafe_btn.setEnabled(self.training_mode)

        status_text = "TRAINING MODE" if self.training_mode else "DETECTION MODE"
        self.status_label.setText(status_text)


    def save_training_sample(self, label):
        """Save current frame's features with manual label"""
        if not hasattr(self, 'current_features'):
            self.status_label.setText("No features to save")
            return

        if len(self.training_data) >= self.training_sample_limit:
            self.status_label.setText("Training limit reached")
            return

        self.training_data.append(self.current_features)
        self.training_labels.append(label)

        # Update UI
        self.training_progress.setText(f"Samples: {len(self.training_data)}/{self.training_sample_limit}")
        self.status_label.setText(f"Saved {len(self.training_data)} samples")

        # Visual feedback
        flash_color = (0, 255, 0) if label == 0 else (0, 0, 255)
        self.video_label.setStyleSheet(f"background-color: rgb{flash_color};")
        QTimer.singleShot(200, lambda: self.video_label.setStyleSheet(""))

    def train_model(self):
        """Train the ML model with collected data"""
        if not self.training_data:
            self.status_label.setText("No training data")
            return

        self.status_label.setText("Training model...")
        QApplication.processEvents()  # Update UI

        try:
            X = np.vstack(self.training_data)
            y = np.array(self.training_labels)

            # Show progress dialog
            progress = QProgressDialog("Training model...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Training")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Train in a separate thread
            self.train_thread = TrainingThread(self.ml_analyzer, X, y)
            self.train_thread.finished.connect(lambda: (
                progress.close(),
                self.status_label.setText("Model trained successfully"),
                self.training_progress.setText("Samples: 0/200")
            ))
            self.train_thread.start()

        except Exception as e:
            self.status_label.setText(f"Training error: {str(e)}")
            if 'progress' in locals():
                progress.close()


    def _train_model_thread(self, X, y):
        """Thread function for model training"""
        try:
            self.ml_analyzer.model.fit(X, y)
            dump(self.ml_analyzer.model, self.config.get('model_path', 'touch_model.joblib'))

            # Clear training data
            self.training_data = []
            self.training_labels = []

            # Update UI via signal or direct call
            self.status_label.setText("Model trained successfully")
            self.training_progress.setText("Samples: 0/200")

        except Exception as e:
            self.status_label.setText(f"Training failed: {str(e)}")


    def update_fps_counter(self):
        """Update FPS counter every second"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.avg_fps = self.frame_count / (current_time - self.last_fps_update)
            self.fps_label.setText(f"FPS: {self.avg_fps:.1f}")
            self.frame_count = 0
            self.last_fps_update = current_time


    def init_detection_modules(self):
        """Initialize all detection and analysis modules"""
        self.hand_detector = HandDetectionModule(
            maxHands=self.config.get('max_hands', 2),
            detectionCon=self.config.get('hand_detection_confidence', 0.7),
            trackCon=self.config.get('hand_tracking_confidence', 0.5)
        )

        self.pose_detector = PoseDetectionModule(
            detectionCon=self.config.get('pose_detection_confidence', 0.7),
            trackCon=self.config.get('pose_tracking_confidence', 0.5)
        )

        self.analyzer = InteractionAnalyzer(
            safe_threshold=self.config.get('alert_threshold', 50)
        )

        self.ml_analyzer = MLInteractionAnalyzer(
            model_path=self.config.get('model_path', 'touch_model.joblib')
        )



    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TouchAware: Good vs Bad Touch Detection")
        self.setWindowIcon(QIcon('assets/icon.png'))
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QHBoxLayout()

        # Left panel - Video display
        video_panel = QVBoxLayout()

        # Video display with frame
        self.video_frame = QGroupBox("Live Detection")
        self.video_frame.setStyleSheet("QGroupBox { border: 2px solid gray; border-radius: 5px; }")
        video_frame_layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        video_frame_layout.addWidget(self.video_label)

        # FPS and status display
        self.status_bar = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.status_label = QLabel("Ready")
        self.mode_label = QLabel("Mode: Detection")

        self.status_bar.addWidget(self.fps_label)
        self.status_bar.addStretch()
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addStretch()
        self.status_bar.addWidget(self.mode_label)

        video_frame_layout.addLayout(self.status_bar)
        self.video_frame.setLayout(video_frame_layout)
        video_panel.addWidget(self.video_frame)

        # Right panel - Controls
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)

        # Input selection
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()

        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.setIcon(QIcon('assets/webcam.png'))
        self.webcam_btn.clicked.connect(self.use_webcam)

        self.video_btn = QPushButton("Open Video File")
        self.video_btn.setIcon(QIcon('assets/video.png'))
        self.video_btn.clicked.connect(self.open_video)

        input_layout.addWidget(self.webcam_btn)
        input_layout.addWidget(self.video_btn)
        input_group.setLayout(input_layout)
        control_panel.addWidget(input_group)

        # Detection settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 100)
        self.sensitivity_slider.setValue(self.config.get('alert_threshold', 50))
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)

        settings_layout.addWidget(QLabel("Sensitivity:"))
        settings_layout.addWidget(self.sensitivity_slider)

        self.privacy_check = QPushButton("Toggle Privacy Blur")
        self.privacy_check.setCheckable(True)
        self.privacy_check.setChecked(self.config.get('privacy_blur', True))
        self.privacy_check.clicked.connect(self.toggle_privacy_blur)


        # For sounds
        self.test_sound_btn = QPushButton("Test Sounds")
        self.test_sound_btn.setIcon(QIcon('assets/speaker.png'))  # Add a speaker icon to your assets folder
        self.test_sound_btn.clicked.connect(self.test_sounds)
        settings_layout.addWidget(self.test_sound_btn)

        settings_group.setLayout(settings_layout)
        control_panel.addWidget(settings_group)


        settings_layout.addWidget(self.privacy_check)
        settings_group.setLayout(settings_layout)
        control_panel.addWidget(settings_group)

        # Machine Learning Panel
        ml_group = QGroupBox("Machine Learning")
        ml_layout = QVBoxLayout()

        self.train_mode_btn = QPushButton("Enable Training Mode")
        self.train_mode_btn.setCheckable(True)
        self.train_mode_btn.clicked.connect(self.toggle_training_mode)

        self.save_safe_btn = QPushButton("Save Safe Sample")
        self.save_safe_btn.setIcon(QIcon('assets/safe.png'))
        self.save_safe_btn.clicked.connect(lambda: self.save_training_sample(0))

        self.save_unsafe_btn = QPushButton("Save Unsafe Sample")
        self.save_unsafe_btn.setIcon(QIcon('assets/unsafe.png'))
        self.save_unsafe_btn.clicked.connect(lambda: self.save_training_sample(2))

        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.setIcon(QIcon('assets/train.png'))
        self.train_model_btn.clicked.connect(self.train_model)

        ml_layout.addWidget(self.train_mode_btn)
        ml_layout.addWidget(self.save_safe_btn)
        ml_layout.addWidget(self.save_unsafe_btn)
        ml_layout.addWidget(self.train_model_btn)

        # Training progress
        self.training_progress = QLabel("Samples: 0/200")
        ml_layout.addWidget(self.training_progress)

        ml_group.setLayout(ml_layout)
        control_panel.addWidget(ml_group)

        # Emergency controls
        emergency_group = QGroupBox("Emergency")
        emergency_layout = QVBoxLayout()

        self.emergency_btn = QPushButton("Trigger Emergency Alert")
        self.emergency_btn.setIcon(QIcon('assets/emergency.png'))
        self.emergency_btn.setStyleSheet("background-color: red; color: white;")
        self.emergency_btn.clicked.connect(lambda: self.send_emergency_alert('manual'))

        emergency_layout.addWidget(self.emergency_btn)
        emergency_group.setLayout(emergency_layout)
        control_panel.addWidget(emergency_group)

        # Add stretch to push everything up
        control_panel.addStretch()

        # Combine main layout
        main_layout.addLayout(video_panel, 70)
        main_layout.addLayout(control_panel, 30)

        self.setLayout(main_layout)

        # Apply styles
        self.apply_styles()




    def use_webcam(self):
        """Start webcam capture"""
        self._start_capture(0)

    def open_video(self):
        """Open video file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self._start_capture(file_path)


    def load_config(self):
        """Load configuration from file or use defaults"""
        try:
            with open('config/app_config.json') as f:
                return json.load(f)
        except:
            return {
                'target_fps': 30,
                'training_sample_limit': 200,
                'privacy_blur': True,
                'alert_threshold': 50,
                'zone_colors': {
                    'groin': [0, 0, 255],
                    'buttocks': [0, 0, 200],
                    'chest': [0, 165, 255],
                    'mouth': [0, 255, 255]
                }
            }


    def toggle_privacy_blur(self, checked):
        """Toggle privacy blur feature"""
        self.config['privacy_blur'] = checked
        status = "ON" if checked else "OFF"
        self.status_label.setText(f"Privacy blur: {status}")


    def update_sensitivity(self, value):
        """Update sensitivity threshold from slider"""
        self.analyzer.safe_threshold = value
        self.config['alert_threshold'] = value
        self.status_label.setText(f"Sensitivity set to {value}")



    def apply_styles(self):
        """Apply consistent styling to the UI"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI';
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton {
                padding: 8px;
                min-width: 100px;
            }
            QLabel {
                padding: 5px;
            }
        """)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        self.video_frame.setFont(title_font)

    def _start_capture(self, source):
        """Start video capture from source with proper initialization"""
        try:
            # Release previous capture if exists
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                time.sleep(0.1)  # Small delay for resource release

            # Initialize new capture
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.status_label.setText("Error opening video source")
                self.cap = None  # Ensure it's set to None if failed
                return

            # Set input mode
            is_webcam = source == 0
            self.hand_detector.set_input_mode(is_webcam)

            # Start timer
            self.status_label.setText("Processing...")
            self.timer.start()

            # Update UI
            source_name = "Webcam" if is_webcam else "Video File"
            self.video_frame.setTitle(f"Live Detection - {source_name}")

        except Exception as e:
            self.status_label.setText(f"Capture error: {str(e)}")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None


    def process_frame(self, img):
        """Process a single frame through the detection pipeline"""
        # Process hands
        if img is None:
            return None

        try:
            img, hands_data = self.hand_detector.multiHandFinder(img, draw=True)
            hands_landmarks = self.hand_detector.multiHandPositionFinder(img, draw=True)

            # Process pose
            img = self.pose_detector.poseFinder(img, draw=True)
            pose_landmarks = self.pose_detector.positionFinder(img, draw=True)

            # Extract body points
            body_points = self._get_body_points(pose_landmarks, img)
            self.body_points = body_points

            # Apply privacy protection if enabled
            if self.config.get('privacy_blur', True):
                img = self.apply_privacy_protection(img)

            # Draw safety zones
            self.draw_safety_zones(img, body_points)

            # Analyze interactions if we have data
            if hands_landmarks and body_points:
                self._analyze_interactions(img, hands_landmarks, body_points)

                # If in training mode, extract features
                if self.training_mode:
                    self.current_features = self.ml_analyzer.extract_features(
                        [(x, y) for _, _, x, y in hands_landmarks[0]],
                        body_points
                    )
                    cv2.putText(img, "TRAINING MODE - Label Current Pose",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)

            return img

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def update_frame(self):
        """Process each video frame with comprehensive safety checks"""
        # Early return if already processing or no video capture initialized
        if self.is_processing or not hasattr(self, 'cap') or self.cap is None:
            return

        self.is_processing = True
        start_time = time.perf_counter()  # More precise timing

        try:
            # Check if capture device is opened
            if not self.cap.isOpened():
                self.timer.stop()
                self.status_label.setText("Video source not available")
                return

            # Read frame with error handling
            success, img = self.cap.read()
            if not success:
                self.timer.stop()
                self.status_label.setText("Video ended" if isinstance(self.cap, cv2.VideoCapture)
                                          else "Error reading frame")
                return

            # Skip processing if frame is empty
            if img is None or img.size == 0:
                print("Warning: Empty frame received")
                return

            # Process frame with additional checks
            processed_img = self.process_frame(img)
            if processed_img is None:
                print("Warning: Frame processing returned None")
                return

            # Display the processed frame
            self.display_image(processed_img)

            # Update FPS counter
            self.update_fps_counter()

        except cv2.error as cv_err:
            print(f"OpenCV error processing frame: {cv_err}")
            self.status_label.setText("Video processing error")
            self.timer.stop()
        except Exception as e:
            print(f"Unexpected error processing frame: {e}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.is_processing = False

            # Only adjust FPS if we successfully processed a frame
            if 'img' in locals() and img is not None:
                processing_time = (time.perf_counter() - start_time) * 1000  # in ms
                target_time = 1000 / self.config.get('target_fps', 30)

                # More sophisticated FPS adjustment
                if processing_time < target_time * 0.7:  # We have significant headroom
                    new_interval = max(5, int(self.timer.interval() * 0.9))  # Gradual increase
                elif processing_time < target_time * 0.9:  # Moderate headroom
                    new_interval = max(5, self.timer.interval() - 1)
                elif processing_time > target_time * 1.5:  # Falling behind
                    new_interval = min(100, self.timer.interval() + 10)  # Significant decrease
                elif processing_time > target_time * 1.1:  # Slightly behind
                    new_interval = min(100, self.timer.interval() + 2)

                self.timer.setInterval(new_interval)





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

            try:
                hand_label = hand[0][0] if hand else 'unknown'
                hand_fingertips = [(x, y) for _, id, x, y in hand if id in [8, 12, 16, 20]]

                if hand_fingertips:
                    zone, distance = self.analyzer.analyze(hand_fingertips, body_points)
                    if zone != "none":
                        current_time = pygame.time.get_ticks()
                        risk_level = 2 if zone in ['groin', 'buttocks'] else 1

                        if current_time - self.last_alert_time > 500:  # 0.5 second cooldown
                            self.play_alert(risk_level)
                            self.log_incident(zone, risk_level)
                            self.last_alert_time = current_time

                        cv2.putText(
                            img, f"{hand_label.title()} hand near {zone.upper()}",
                            (10, 60 if hand_label == 'left' else 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                        )
            except Exception as e:
                print(f"Error analyzing hand interaction: {e}")


    def display_image(self, img):

        if img is None:
            return

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"Error displaying image: {str(e)}")

    def closeEvent(self, event):
        """Handle application shutdown safely"""
        try:
            # Stop the timer first
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()

            # Release video capture if it exists
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()

            # Clean up pygame mixer
            if pygame.mixer.get_init():
                pygame.mixer.quit()

            # Close any open training progress dialogs
            if hasattr(self, 'train_thread') and self.train_thread.isRunning():
                self.train_thread.terminate()

        except Exception as e:
            print(f"Error during shutdown: {e}")

        # Ensure the event is accepted
        event.accept()

    def init_audio_system(self):
        """Robust audio initialization with fallbacks"""
        try:
            # Initialize mixer with proper parameters
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

            # Verify sound files exist
            if not all([os.path.exists(f"Sounds/{sound}") for sound in ['alert.wav', 'warning.wav']]):
                raise FileNotFoundError("Sound files missing")

            # Pre-load sounds with verification
            self.alert_sound = pygame.mixer.Sound("Sounds/alert.wav")
            self.warning_sound = pygame.mixer.Sound("Sounds/warning.wav")

            # Create dedicated channels
            self.alert_channel = pygame.mixer.Channel(0)
            self.warning_channel = pygame.mixer.Channel(1)

            # Test sound loading
            if not all([self.alert_sound, self.warning_sound]):
                raise ValueError("Sound objects not created")

            self.use_system_sounds = False
            print("Audio system initialized successfully")

        except Exception as e:
            print(f"Audio init failed: {e}. Using system sounds.")
            self.use_system_sounds = True
            # Initialize pygame mixer anyway for system sounds
            try:
                pygame.mixer.init()
            except:
                pass

    def play_alert(self, level):
        """Level: 1 (warning), 2 (emergency)"""
        print(f"Attempting to play level {level} alert")  # Debug

        try:
            if self.use_system_sounds:
                print("Using system beep as fallback")
                freq = 800 if level == 1 else 2000
                duration = 500  # ms
                winsound.Beep(freq, duration)
            else:
                sound = self.warning_sound if level == 1 else self.alert_sound
                channel = self.warning_channel if level == 1 else self.alert_channel

                if not channel.get_busy():
                    print(f"Playing {level} sound on channel {channel}")
                    channel.play(sound)
                else:
                    print(f"Channel {channel} busy, skipping play")

        except Exception as e:
            print(f"Error in play_alert: {e}")
            # Final fallback
            try:
                winsound.Beep(1000, 300)
            except:
                print("Couldn't use any sound system")

    def test_sounds(self):
        """Test all alert sounds"""
        self.play_alert(1)  # Should play warning
        time.sleep(0.5)
        self.play_alert(2)  # Should play alert

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

    # Set application style
    app.setStyle('Fusion')

    window = TouchDetectionApp()
    window.show()
    sys.exit(app.exec_())