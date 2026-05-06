import cv2
import numpy as np
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

    def get_left_thumb_pos(self):
        """Returns the (x, y) coordinates of the right index finger tip (landmark 8)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[4]
                    return landmark.x, landmark.y
        return None

    def get_joint_angles(self):
        """
        Returns a 30-element list of joint angles (in degrees) for both hands.
        The first 15 elements are for the Left hand (user's perspective),
        and the next 15 are for the Right hand. Missing hands are padded with 0.0.
        """
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            left_angles = [0.0] * 15
            right_angles = [0.0] * 15
            
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                label = handedness.classification[0].label # 'Left' or 'Right'
                landmark_list = self.hand_results.multi_hand_landmarks[i].landmark
                
                joint = np.zeros((21, 3))
                for j, lm in enumerate(landmark_list):
                    joint[j] = [lm.x, lm.y, lm.z]
                    
                # Compute vectors between joints
                v1_idx = np.array([0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18])
                v2_idx = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
                v3_idx = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])

                v1 = joint[v2_idx] - joint[v1_idx]
                v2 = joint[v3_idx] - joint[v2_idx]
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1, axis=1)
                v2_norm = np.linalg.norm(v2, axis=1)
                # handle division by zero safely
                v1_norm[v1_norm == 0] = 1e-6
                v2_norm[v2_norm == 0] = 1e-6
                v1 = v1 / v1_norm[:, np.newaxis]
                v2 = v2 / v2_norm[:, np.newaxis]
                
                # Calculate angle using dot product
                dot_product = np.sum(v1 * v2, axis=1)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angles = np.arccos(dot_product) * (180.0 / np.pi)
                angles_list = np.round(angles, 2).tolist()
                
                # Assign to correct hand corresponding to the user's view
                if label == 'Right': # 사람 기준 왼손
                    left_angles = angles_list
                else: # 사람 기준 오른손
                    right_angles = angles_list
                    
            return left_angles + right_angles
        return None
