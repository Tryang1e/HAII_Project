import cv2
import numpy as np
import mediapipe.python.solutions.hands as mp_hands

class Detector:
    def __init__(self):
        # Hands for finger joint recognition (Primary)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.hand_results = None
        self.width = 0
        self.height = 0
        
        # EMA(지수 이동 평균) 기반 관절 떨림(Jitter) 보정
        self.ema_alpha = 0.7  # 평활화 계수 (지연 시간/Lag을 없애기 위해 0.4 -> 0.7로 반응성 대폭 향상)
        self.lm_history = {'Left': None, 'Right': None}

    def process_hands(self, frame):
        self.height, self.width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.hand_results = self.hands.process(rgb)
        
        # 관절 좌표 자체에 EMA 스무딩 적용 (ui.py, app.py, mouse.py 모든 곳에 보정 효과 적용)
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            current_labels = set()
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                label = handedness.classification[0].label
                current_labels.add(label)
                landmarks = self.hand_results.multi_hand_landmarks[i].landmark
                
                if self.lm_history[label] is None:
                    # 초기 발견 시 히스토리 기록
                    self.lm_history[label] = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                else:
                    # 이전 좌표와 현재 좌표를 alpha 비율로 혼합 (EMA)
                    raw_arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    smoothed_arr = self.lm_history[label] * (1 - self.ema_alpha) + raw_arr * self.ema_alpha
                    self.lm_history[label] = smoothed_arr
                    
                    # MediaPipe Landmark 객체에 다시 덮어쓰기
                    for j, lm in enumerate(landmarks):
                        lm.x, lm.y, lm.z = smoothed_arr[j][0], smoothed_arr[j][1], smoothed_arr[j][2]
            
            # 화면에서 사라진 손의 히스토리 삭제
            for label in ['Left', 'Right']:
                if label not in current_labels:
                    self.lm_history[label] = None
        else:
            self.lm_history['Left'] = None
            self.lm_history['Right'] = None
            
        return self.hand_results.multi_hand_landmarks is not None

    def _get_corrected_hands(self):
        """
        MediaPipe의 손 방향 예측이 두 손일 때 교차되거나 흔들리는 오류를 방지하기 위해,
        화면상의 X 좌표(좌/우 위치)를 기준으로 손의 방향(Left/Right)을 보정하여 반환합니다.
        (display_image는 좌우 반전된 상태이므로, x가 작은 쪽이 사용자 왼손(Mediapipe 'Right'),
        x가 큰 쪽이 사용자 오른손(Mediapipe 'Left')입니다.)
        """
        if not (self.hand_results and self.hand_results.multi_hand_landmarks):
            return []
            
        hands_data = []
        for i, handedness in enumerate(self.hand_results.multi_handedness):
            label = handedness.classification[0].label
            
            landmarks = self.hand_results.multi_hand_landmarks[i].landmark
            
            # 손바닥 중심(palm center) 계산
            palm_indices = [0, 5, 9, 13, 17]
            avg_x = float(sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices))
            avg_y = float(sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices))
            
            hands_data.append({
                'label': label,
                'x': avg_x,
                'y': avg_y,
                'landmarks': landmarks
            })
            
        # 화면에 두 손이 모두 보일 때의 위치 기반 강제 보정을 제거합니다.
        # 이 강제 보정 때문에 거울 모드 환경에서 두 손이 등장하는 순간 좌우가 바뀌는 치명적인 버그가 발생했습니다.
        return hands_data

    def get_hand_info(self, hand_label):
        """
        Returns info for a hand. 
        hand_label: 'Left' or 'Right' (user's perspective)
        """
        target_label = 'Right' if hand_label == 'Left' else 'Left'
        hands_data = self._get_corrected_hands()
        
        for hand in hands_data:
            if hand['label'] == target_label:
                landmarks = hand['landmarks']
                return {
                    'pos': (hand['x'], hand['y']),
                    'middle': (float(landmarks[12].x), float(landmarks[12].y), float(landmarks[12].z)),
                    'thumb': (float(landmarks[4].x), float(landmarks[4].y), float(landmarks[4].z)),
                    'index': (float(landmarks[8].x), float(landmarks[8].y), float(landmarks[8].z)),
                    'landmarks': landmarks
                }
        return None

    def is_hand_fist(self, hand_label):
        """Checks if the specified hand is a fist."""
        target_label = 'Right' if hand_label == 'Left' else 'Left'
        hands_data = self._get_corrected_hands()
        
        for hand in hands_data:
            if hand['label'] == target_label:
                landmarks = hand['landmarks']
                
                # 주먹 판정: 손가락 끝(tip)이 손가락 뿌리(mcp)보다 손목(0)에 더 가까운지 확인
                tips = [8, 12, 16, 20]
                mcps = [5, 9, 13, 17]
                
                folded_count = 0
                for tip, mcp in zip(tips, mcps):
                    dist_tip = ((landmarks[tip].x - landmarks[0].x)**2 + (landmarks[tip].y - landmarks[0].y)**2)**0.5
                    dist_mcp = ((landmarks[mcp].x - landmarks[0].x)**2 + (landmarks[mcp].y - landmarks[0].y)**2)**0.5
                    if dist_tip < dist_mcp:
                        folded_count += 1
                
                return folded_count >= 3
        return False

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
