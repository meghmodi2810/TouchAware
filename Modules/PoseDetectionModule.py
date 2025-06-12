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
