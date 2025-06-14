import cv2
import mediapipe as mp

class PoseDetectionModule:
    def __init__(self, staticMode=False, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode,
                                     model_complexity=self.modelComplexity,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def poseFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return image

    def positionFinder(self, image, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            h, w, _ = image.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    # In PoseDetectionModule.py
    def get_sensitive_zones(self, pose_landmarks, img_shape):
        h, w = img_shape[:2]
        zones = {}

        # Groin (between hips)
        if len(pose_landmarks) >= 25:
            left_hip = pose_landmarks[23][1:]  # ID 23
            right_hip = pose_landmarks[24][1:]  # ID 24
            zones['groin'] = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2 + int(0.05 * h)
            )

            # Buttocks (below hips)
            zones['buttocks'] = (
                zones['groin'][0],
                zones['groin'][1] + int(0.15 * h)  # 15% below groin
            )

            # Chest (between shoulders with vertical offset)
            left_shoulder = pose_landmarks[11][1:]
            right_shoulder = pose_landmarks[12][1:]
            zones['chest'] = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2 + int(0.1 * h)
            )

            # Mouth (approximation using nose and upper chest)
            nose = pose_landmarks[0][1:]
            zones['mouth'] = (
                nose[0],
                nose[1] + int(0.05 * h)
            )


        # Lips (using face landmarks if available)
        if hasattr(self, 'face_landmarks'):
            zones['lips'] = self.face_landmarks[0][1:]  # Simplified

        return zones