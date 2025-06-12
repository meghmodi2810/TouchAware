import cv2
import mediapipe as mp


class HandDetectionModule:
    finger_tips_ids = [4, 8, 12, 16, 20]

    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.is_webcam = False

        self.mpHands = mp.solutions.hands
        self.Hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def set_input_mode(self, is_webcam):
        self.is_webcam = is_webcam


    def handFinder(self, image, draw = True, ):
        if self.is_webcam:
            image = cv2.flip(image, 1)

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageRGB.flags.writeable = False
        self.results = self.Hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return image

    def multiHandFinder(self, image, draw = True):
        if self.is_webcam:
            image = cv2.flip(image, 1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageRGB.flags.writeable = False
        self.results = self.Hands.process(imageRGB)

        hands_data = []  # List to hold data for each hand

        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                hands_data.append({'label': label, 'landmarks': hand_landmarks})

        return image, hands_data

    def positionFinder(self, image, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            particularHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(particularHand.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def multiHandPositionFinder(self, image, draw=True):
        hands_landmarks = []

        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()  # 'left' or 'right'
                lmList = []

                for id, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    lmList.append([hand_label, id, cx, cy])

                    if draw:
                        cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                hands_landmarks.append(lmList)

        return hands_landmarks

    def fingerName(self, lmList):
        fingers = []

        if len(lmList) < 21:
            return fingers


        if lmList[self.finger_tips_ids[0]][1] > lmList[self.finger_tips_ids[0] - 1][1]:
            fingers.append("Thumb")


        for i, tip_id in enumerate(self.finger_tips_ids[1:], start=1):
            if lmList[tip_id][2] < lmList[tip_id - 2][2]:
                fingers.append(["Index", "Middle", "Ring", "Pinky"][i - 1])
        return fingers

    def multiFingerName(self, lmList):
        fingers = []

        if len(lmList) < 21:
            return fingers

        # Extract only x, y by index for readability
        id_to_coords = {id: (x, y) for _, id, x, y in lmList}

        # Thumb
        if id_to_coords[4][0] > id_to_coords[3][0]:  # Assuming right hand
            fingers.append("Thumb")

        # Fingers (Index to Pinky)
        for i, tip_id in enumerate(self.finger_tips_ids[1:], start=1):
            if id_to_coords[tip_id][1] < id_to_coords[tip_id - 2][1]:
                fingers.append(["Index", "Middle", "Ring", "Pinky"][i - 1])

        return fingers

    def fingersBool(self, lmList):
        fingers_status = [False] * 5

        if len(lmList) < 21:
            return fingers_status

        if lmList[4][1] > lmList[3][1]:
            fingers_status[0] = True

        for i in range(1, 5):
            if lmList[self.finger_tips_ids[i]][2] < lmList[self.finger_tips_ids[i] - 2][2]:
                fingers_status[i] = True

        return fingers_status
