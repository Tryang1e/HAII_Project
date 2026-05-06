import cv2
import mediapipe.python.solutions.hands as mp_hands

class Detector:
    def __init__(self):
        # Hands for finger joint recognition (Primary)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            model_complexity=0
        )
        self.hand_results = None
        self.width = 0
        self.height = 0

    def process_hands(self, frame):
        self.height, self.width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.hand_results = self.hands.process(rgb)
        return self.hand_results.multi_hand_landmarks is not None

    # 사람 기준 왼손 (카메라 기준 Right 레이블)
    def get_left_hand_pos(self):
        """Returns the (x, y) coordinates of the hand palm centroid using wrist and MCP joints."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                
                if handedness.classification[0].label == 'Right':
                    landmarks = self.hand_results.multi_hand_landmarks[i].landmark
                    # 손바닥 중심을 결정하는 주요 관절 (손목 0, 검지~새끼 기저부 5, 9, 13, 17)
                    palm_indices = [0, 5, 9, 13, 17]
                    avg_x = sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices)
                    avg_y = sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices)
                    return avg_x, avg_y
        return None

    def get_left_index_pos(self):
        """Returns the (x, y) coordinates of the left index finger tip (landmark 8)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[8]
                    return landmark.x, landmark.y
        return None

    # 사람 기준 오른손 (카메라 기준 Left 레이블)
    def get_right_hand_pos(self):
        """Returns the (x, y) coordinates of the hand palm centroid using wrist and MCP joints."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    landmarks = self.hand_results.multi_hand_landmarks[i].landmark
                    # 손바닥 중심을 결정하는 주요 관절 (손목 0, 검지~새끼 기저부 5, 9, 13, 17)
                    palm_indices = [0, 5, 9, 13, 17]
                    avg_x = sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices)
                    avg_y = sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices)
                    return avg_x, avg_y
        return None

    def get_right_index_pos(self):
        """Returns the (x, y) coordinates of the right index finger tip (landmark 8)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Right':   # 오른손 엄지로 변경
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[4]
                    return landmark.x, landmark.y
        return None

    def get_thumb_pos(self, is_left=True):
        """Returns the (x, y) coordinates of the thumb tip (landmark 4)."""
        label = 'Right' if is_left else 'Left'
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == label:
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[4]
                    return landmark.x, landmark.y
        return None

    def get_index_pos(self, is_left=True):
        """Returns the (x, y) coordinates of the index finger tip (landmark 8)."""
        label = 'Right' if is_left else 'Left'
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == label:
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[8]
                    return landmark.x, landmark.y
        return None

