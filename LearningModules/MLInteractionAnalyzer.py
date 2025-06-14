import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import pygame
from PyQt5.QtCore import QThread

class MLInteractionAnalyzer:
    def __init__(self, model_path='touch_model.joblib'):
        # Body zones and their safety levels (0=safe, 1=caution, 2=unsafe)
        self.zone_classes = {
            'shoulder': 0,
            'hand': 0,
            'chest': 2,
            'waist': 2,
            'thigh': 1,
            'groin': 3,
            'buttocks': 3,
            'lips': 2
        }

        # Initialize ML model
        self.model = self._init_model()
        self.scaler = None
        self.model_path = model_path
        self.model = self._load_or_create_model()

    def _load_or_create_model(self):
        try:
            return load(self.model_path)
        except:
            print("No saved model found. Creating new model.")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100)

    def save_model(self):
        """Save the current model to disk"""
        dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")


    def _init_model(self):
        try:
            return load('touch_model.joblib')  # Load pre-trained model
        except:
            # Fallback model if no saved model exists
            return RandomForestClassifier(n_estimators=100)

    def extract_features(self, hand_points, body_points):
        """Convert raw points into ML features"""
        features = []

        # 1. Minimum distance to sensitive zones
        for zone, point in body_points.items():
            min_dist = min([self.euclidean_distance(hand, point) for hand in hand_points])
            features.append(min_dist)

        # 2. Hand velocity (if tracking multiple frames)
        if hasattr(self, 'last_position'):
            velocity = self.euclidean_distance(hand_points[0], self.last_position)
            features.append(velocity)
        self.last_position = hand_points[0]

        # 3. Hand orientation and spread
        if len(hand_points) >= 5:  # If we have multiple finger points
            spread = np.std([p[0] for p in hand_points])  # Horizontal spread
            features.append(spread)

        return np.array(features).reshape(1, -1)

    def analyze(self, hand_points, body_points):
        """Enhanced ML-powered analysis"""
        if not body_points:
            return "none", 0

        # Extract features
        X = self.extract_features(hand_points, body_points)

        # Standardize features if scaler exists
        if self.scaler:
            X = self.scaler.transform(X)

        # Predict danger level (0-2)
        danger_level = self.model.predict(X)[0]

        # Find closest zone
        closest_zone, min_dist = self._closest_zone(hand_points, body_points)

        # Determine if touch is bad
        if danger_level >= 1 and min_dist < 50:  # 50 pixels threshold
            self._trigger_alert(closest_zone, danger_level)
            return closest_zone, danger_level
        return "none", 0

    def _trigger_alert(self, zone, level):
        # Sound alert
        pygame.mixer.Sound("alerts/high_alert.wav" if level == 2 else "alerts/warning.wav").play()

        # Visual feedback (could be integrated with GUI)
        print(f"ALERT LEVEL {level}: Inappropriate contact detected near {zone}")

    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def train_model(self):
        """Train the ML model with collected data"""


        if not self.training_data:
            print("No training data! Save samples first.")
            return


        # if len(self.training_data) == 0:
        #     print("No training data available!")
        #     return

        X = np.vstack(self.training_data)
        y = np.array(self.training_labels)
        self.thread = TrainingThread(self.ml_analyzer, X, y)
        self.thread.start()  # Non-blocking
        # Train model
        self.ml_analyzer.model.fit(X, y)

        # Save model
        self.ml_analyzer.save_model()

        # Clear training data
        self.training_data = []
        self.training_labels = []
        print("Model trained on", len(X), "samples!")

class TrainingThread(QThread):
    def __init__(self, analyzer, X, y):
        super().__init__()
        self.analyzer = analyzer
        self.X = X
        self.y = y

    def run(self):
        self.analyzer.model.fit(self.X, self.y)
        self.analyzer.save_model()
