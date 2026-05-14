import os
import cv2
import time
import numpy as np
import json
from pathlib import Path
from collections import deque
from core.camera import get_camera
from core.mouse import VirtualMouse
from core.ui import UIManager
from core.kiosk import KioskUI
from detector.landmark import Detector
from pykalman import KalmanFilter
from enum import Enum

class HandState(Enum):
    IDLE = 1
    DRAWING = 2
    MENU_SELECTION = 3
class InteractionApp:
    def __init__(self):
        # --- [수정] 카메라 인덱스 자동 검색 지원 ---
        # 환경 변수가 있으면 사용하고, 없으면 None을 넘겨 get_camera가 자동으로 찾게 함
        camera_index = os.environ.get('CAMERA_INDEX')
        if camera_index:
            camera_index = int(camera_index)
        else:
            camera_index = None # 자동 검색 모드

        self.camera = get_camera(width=1280, height=720, fps=60, camera_index=camera_index)

        
        self.mouse = VirtualMouse()
        self.detector = Detector()
        self.ui = UIManager()
        self.kiosk = KioskUI()
        
        self.all_strokes = []  
        self.active_stroke = []
        self.undone_strokes = []
        self.current_state = HandState.IDLE
        
        self.view_rot_x = 0  
        self.view_rot_y = 0  
        self.view_zoom = 1.0
        self.prev_rot_pos = None
        self.prev_zoom_y = None

        # --- [Kalman filter 기반 보정 관련] ---
        dt = 1.0
        transition_matrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        observation_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        self.kf_draw = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=np.zeros(6),
            initial_state_covariance=np.eye(6),
            # 0~1 좌표계에 맞는 스케일로 파라미터 대폭 수정 (기존 0.2는 오차가 화면의 20%라고 가정한 것이라 엄청난 딜레이 발생)
            observation_covariance=np.eye(3) * 0.0005,  # 측정 노이즈 (반응성을 위해 대폭 낮춤)
            transition_covariance=np.eye(6) * 0.0001   # 상태 전이 노이즈
        )
        self.draw_state_mean = np.zeros(6)
        self.draw_state_cov = np.eye(6)
        self.sx, self.sy, self.sz = 0, 0, 0

        # --- [제스처 초기화 관련] ---
        self.clear_timer_start = None
        self.time_to_clear = 1.0 # 1초 유지 시 초기화

        # --- [프레임 드랍 보정] ---
        self.right_hand_lost_frames = 0
        self.pinch_lost_frames = 0

        # --- [SVM Model Initialization] ---
        self.svm = cv2.ml.SVM_load('virtual-touch-click/gesture_svm_model.xml')
        with open('virtual-touch-click/gesture_labels.json', 'r') as f:
            label_map_str = json.load(f)
            self.int_to_label = {int(k): v for k, v in label_map_str.items()}
        self.angle_history = deque(maxlen=20)
        self.predicted_gesture = None

    def extract_features(self, data_rows, target_frames=20):
        angles = np.array(data_rows, dtype=np.float32)
        num_frames = angles.shape[0]
        num_features = angles.shape[1]
        
        if num_frames == target_frames:
            resampled = angles
        else:
            resampled = np.zeros((target_frames, num_features), dtype=np.float32)
            original_indices = np.linspace(0, 1, num_frames)
            target_indices = np.linspace(0, 1, target_frames)
            for i in range(num_features):
                resampled[:, i] = np.interp(target_indices, original_indices, angles[:, i])
                
        return resampled.flatten()

    def start(self):
        self.camera.start()
        try:
            self.run_loop()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

    def run_loop(self):
        while True:
            frame = self.camera.get_frame()
            if frame is None: continue
            
            display_image = cv2.flip(frame, 1)
            h_img, w_img = display_image.shape[:2]

            self.detector.process_hands(display_image)
            self.ui.draw_landmarks(display_image, self.detector, w_img, h_img)

            angles = self.detector.get_joint_angles()
            if angles is not None:
                self.angle_history.append(angles)
            
            if len(self.angle_history) >= 5:
                features = self.extract_features(list(self.angle_history))
                features_np = np.array([features], dtype=np.float32)
                _, result = self.svm.predict(features_np)
                pred_int = int(result[0][0])
                self.predicted_gesture = self.int_to_label.get(pred_int, None)
            else:
                self.predicted_gesture = None

            user_left = self.detector.get_hand_info('Right')
            user_right = self.detector.get_hand_info('Left')

            # 양손 주먹 판별을 위해 오른손 상태 미리 확인
            is_right_fist = False
            if user_right:
                lm_r = user_right['landmarks']
                def calc_dist_r_early(idx1, idx2):
                    return ((lm_r[idx1].x - lm_r[idx2].x)**2 + (lm_r[idx1].y - lm_r[idx2].y)**2 + (lm_r[idx1].z - lm_r[idx2].z)**2)**0.5
                is_right_fist = (calc_dist_r_early(8, 0) < calc_dist_r_early(5, 0) * 1.1) and (calc_dist_r_early(12, 0) < calc_dist_r_early(9, 0) * 1.1)

            both_fists = False

            # 1. 사용자 왼손 (회전 및 줌)
            left_interacting = False
            if user_left:
                lx, ly = user_left['pos']
                lm = user_left['landmarks']
                
                # X, Y, Z 3축 고려 유클리드 거리 계산 헬퍼 함수 (각도에 따른 왜곡 방지)
                def calc_dist(idx1, idx2):
                    return ((lm[idx1].x - lm[idx2].x)**2 + (lm[idx1].y - lm[idx2].y)**2 + (lm[idx1].z - lm[idx2].z)**2)**0.5
                
                # 제스처 특징점 추출
                dist_thumb_mid = calc_dist(4, 12)
                dist_thumb_idx = calc_dist(4, 8)
                # 손 크기(스케일)에 비례하는 동적 임계값 생성 (거리에 따른 오작동 방지)
                palm_size = calc_dist(0, 9)
                if palm_size < 0.01: palm_size = 0.01
                pinch_th = palm_size * 0.35      # 핀치 시작 임계값 (너그럽던 것을 0.50 -> 0.35로 엄격하게 수정)
                release_th = palm_size * 0.45    # 핀치 해제 임계값
                clear_th = palm_size * 0.70      # 확실히 폈다고 인정할 임계값 (동작 꼬임 방지용)
                
                # 왼손 검지가 펴져 있는지 확실하게 판별 (Tip-Wrist 거리가 MCP-Wrist 거리의 1.3배 이상)
                idx_extended = calc_dist(8, 0) > calc_dist(5, 0) * 1.3
                mid_extended = calc_dist(12, 0) > calc_dist(9, 0) * 1.3
                is_left_open = idx_extended and mid_extended
                
                # 손가락 접힘 판별
                idx_tucked = calc_dist(8, 0) < calc_dist(5, 0) * 1.1
                mid_tucked = calc_dist(12, 0) < calc_dist(9, 0) * 1.1
                ring_tucked = calc_dist(16, 0) < calc_dist(13, 0) * 1.1
                pinky_tucked = calc_dist(20, 0) < calc_dist(17, 0) * 1.1
                
                # 확실히 접힌 상태(완전한 주먹/포인팅 상태) 판별 (O모양 핀치와 구분하기 위함)
                idx_tightly_tucked = calc_dist(8, 0) < calc_dist(5, 0) * 0.9
                mid_tightly_tucked = calc_dist(12, 0) < calc_dist(9, 0) * 0.9
                
                # 주먹(Rotation) 판별: 꽉 쥐지 않고 '느슨한 주먹'만 쥐어도 인식되도록 임계값 대폭 완화 (0.95 -> 1.1)
                is_left_fist = idx_tucked and mid_tucked and ring_tucked and pinky_tucked
                both_fists = is_left_fist and is_right_fist
                
                # 주먹 잠금용 (손목 각도가 틀어져도 풀리지 않도록 매우 넉넉한 임계값 적용: 1.3 -> 1.6)
                idx_lock_tucked = calc_dist(8, 0) < calc_dist(5, 0) * 1.6
                mid_lock_tucked = calc_dist(12, 0) < calc_dist(9, 0) * 1.6

                # 오른손이 드로잉 중일 때는 왼손이 뷰에 들어올 때 발생하는 초기 랜드마크 튐 현상(오입력)을 무시합니다.
                # 양손이 모두 주먹일 때는 모든 동작 무시
                can_left_interact = (self.current_state != HandState.DRAWING) and not both_fists

                # 손가락 관절 중 하나라도 카메라 시야의 극단적 가장자리(2% 이내)에 있으면 
                # 화면 밖으로 나간 것으로 간주하여 MediaPipe의 환각(Hallucination) 핀치를 차단
                is_idx_in_bounds = all((0.02 < lm[i].x < 0.98) and (0.02 < lm[i].y < 0.98) for i in [5, 6, 7, 8])
                is_mid_in_bounds = all((0.02 < lm[i].x < 0.98) and (0.02 < lm[i].y < 0.98) for i in [9, 10, 11, 12])

                # 확실한 핀치 형태인지 수학적으로 검증 (각도 왜곡 방지용: 끝마디 길이의 1.5배 이내)
                is_ui_pinch_strict = is_idx_in_bounds and (dist_thumb_idx < pinch_th) and (dist_thumb_idx < calc_dist(8, 7) * 1.5)
                is_zoom_pinch_strict = is_mid_in_bounds and (dist_thumb_mid < pinch_th) and (dist_thumb_mid < calc_dist(12, 11) * 1.5)

                # zoom 동작: 엄지-중지 핀치 시 약지와 새끼손가락은 접혀 있어야 함
                # 포인팅 상태(중지가 완전 접힘)에서 줌이 발동하는 것을 막기 위해 not mid_tightly_tucked 조건 추가
                is_zoom_heuristic = is_zoom_pinch_strict and (dist_thumb_mid < dist_thumb_idx * 0.8) and ring_tucked and pinky_tucked and not mid_tightly_tucked
                is_zoom_intent = can_left_interact and is_mid_in_bounds and not is_left_open and (dist_thumb_mid < dist_thumb_idx * 0.9) and (is_zoom_heuristic or (self.predicted_gesture == 'zoom' and dist_thumb_mid < palm_size * 0.8 and ring_tucked and pinky_tucked and not mid_tightly_tucked))
                
                # 화면 도구 선택(클릭): 엄지-검지 핀치 외 나머지 손가락은 접혀 있어야 함
                # 주먹 상태(검지가 완전 접힘)에서 클릭이 발동하는 것을 막기 위해 not idx_tightly_tucked 조건 추가
                is_ui_select_heuristic = is_ui_pinch_strict and (dist_thumb_idx < dist_thumb_mid * 0.8) and mid_tucked and ring_tucked and pinky_tucked and not idx_tightly_tucked
                is_ui_select_intent = can_left_interact and is_idx_in_bounds and not is_left_open and (dist_thumb_idx < dist_thumb_mid * 0.9) and (is_ui_select_heuristic or (self.predicted_gesture == 'menu_select' and dist_thumb_idx < palm_size * 0.8 and mid_tucked and ring_tucked and pinky_tucked and not idx_tightly_tucked))

                # rotation 동작: 확실한 핀치 상태가 아닐 때만 주먹으로 인정 (핀치 중 각도 변경으로 인한 오작동 완벽 차단)
                is_rotation_heuristic = is_left_fist and not (is_ui_pinch_strict or is_zoom_pinch_strict)
                is_rotation_intent = can_left_interact and not is_right_fist and not (is_ui_pinch_strict or is_zoom_pinch_strict) and (is_rotation_heuristic or (self.predicted_gesture == 'rotation' and idx_tucked and mid_tucked))
                
                # 포인터 이동: 기존 휴리스틱 기반
                is_pointer_intent = can_left_interact and is_idx_in_bounds and idx_extended and mid_tucked and not is_rotation_heuristic
                
                # --- [동작 상태 잠금 (Hysteresis Lock) 및 우선순위 결정] ---
                is_zoom = False
                is_rotation = False
                is_ui_select = False
                is_pointer = False
                
                # 1순위: Rotation (주먹 상태가 가장 우선순위가 높으며, 다른 상태의 Lock을 무시하고 강제 진입/유지)
                if is_rotation_intent or (self.prev_rot_pos is not None and idx_lock_tucked and mid_lock_tucked and not is_right_fist):
                    is_rotation = True
                # 2순위: Zoom Lock (의식하고 풀어야 할 정도로 널널했던 임계값을 1.5배 -> 1.1배로 낮춤)
                elif self.prev_zoom_y is not None and dist_thumb_mid < release_th * 1.1:
                    is_zoom = True
                # 3순위: UI Select(Pinch) Lock (마찬가지로 1.5배 -> 1.1배로 낮춤)
                elif self.current_state == HandState.MENU_SELECTION and dist_thumb_idx < release_th * 1.1:
                    is_ui_select = True
                # 4순위: 현재 프레임의 의도 반영
                else:
                    is_zoom = is_zoom_intent
                    is_ui_select = is_ui_select_intent
                    
                if not is_zoom and not is_rotation and not is_ui_select:
                    is_pointer = is_pointer_intent
                    
                # 동작 종료 시 상태 변수 초기화 -> 다음 프레임에서 잠금이 오작동하지 않게 하기 위함
                if not is_zoom: self.prev_zoom_y = None
                if not is_rotation: self.prev_rot_pos = None
                if not is_ui_select and self.current_state == HandState.MENU_SELECTION:
                    self.current_state = HandState.IDLE
                
                # 포인터 시각화 (왼손 검지)
                pointer_lx, pointer_ly = int(lm[8].x * w_img), int(lm[8].y * h_img)
                cv2.circle(display_image, (pointer_lx, pointer_ly), 6, (255, 0, 255), 2)
                
                if is_zoom:
                    left_interacting = True
                    if self.prev_zoom_y is None: self.prev_zoom_y = ly
                    dy = ly - self.prev_zoom_y
                    self.view_zoom = np.clip(self.view_zoom - dy * 2.0, 0.1, 5.0)
                    self.prev_zoom_y = ly
                    cv2.putText(display_image, f"ZOOM MODE: {self.view_zoom:.2f}x", (50, 150), 0, 1, (0, 255, 255), 2)
                    cv2.line(display_image, (int(lm[12].x*w_img), int(lm[12].y*h_img)), (int(lm[4].x*w_img), int(lm[4].y*h_img)), (0, 255, 255), 3)
                elif is_rotation:
                    left_interacting = True
                    if self.prev_rot_pos is None: self.prev_rot_pos = (lx, ly)
                    dx, dy = lx - self.prev_rot_pos[0], ly - self.prev_rot_pos[1]
                    self.view_rot_y += dx * 5.0
                    self.view_rot_x -= dy * 5.0
                    self.prev_rot_pos = (lx, ly)
                    cv2.putText(display_image, "ROTATION MODE", (50, 150), 0, 1, (255, 255, 0), 2)
                elif is_pointer or is_ui_select:
                    left_interacting = True
                    self.current_state = HandState.MENU_SELECTION
                    
                    # 기획서 명세에 따라 튀는 현상 방지를 위해 손바닥 무게중심(lx, ly)을 사용하여 마우스 이동
                    self.mouse.move(lx, ly)
                    
                    if is_ui_select:
                        cv2.putText(display_image, "UI SELECT (CLICK)", (50, 150), 0, 1, (255, 0, 255), 2)
                        cv2.circle(display_image, (pointer_lx, pointer_ly), 8, (0, 0, 255), -1) # 클릭 시 빨간색
                        self.mouse.click()
                    else:
                        cv2.circle(display_image, (pointer_lx, pointer_ly), 6, (255, 0, 255), -1)
                else:
                    if self.current_state == HandState.MENU_SELECTION:
                        self.current_state = HandState.IDLE
                    self.prev_rot_pos, self.prev_zoom_y = None, None
            else:
                # 왼손이 화면 밖으로 완전히 나갔을 때 상태 초기화
                if self.current_state == HandState.MENU_SELECTION:
                    self.current_state = HandState.IDLE
                self.prev_rot_pos, self.prev_zoom_y = None, None

            # 2. 사용자 오른손 (그리기 전용)
            if user_right:
                self.right_hand_lost_frames = 0
                hx, hy = user_right['pos']
                idx_pos, thumb_pos = user_right['index'], user_right['thumb']
                
                # 왼손이 어떠한 조작(줌, 회전, 메뉴 선택 등)도 하지 않을 때만 마우스 포인팅은 왼손 전용이므로 오른손 마우스 제어 삭제
                
                dist_3d = ((idx_pos[0] - thumb_pos[0])**2 + (idx_pos[1] - thumb_pos[1])**2 + (idx_pos[2] - thumb_pos[2])**2)**0.5
                lm = user_right['landmarks']
                
                # 3D 유클리드 거리 계산 (오른손, X,Y,Z 좌표를 전부 고려하여 각도에 따른 왜곡 방지)
                def calc_dist_rh(idx1, idx2):
                    return ((lm[idx1].x - lm[idx2].x)**2 + (lm[idx1].y - lm[idx2].y)**2 + (lm[idx1].z - lm[idx2].z)**2)**0.5
                    
                hand_scale = calc_dist_rh(0, 5)
                if hand_scale < 0.01: hand_scale = 0.01
                # 드로잉 의도 임계값을 더욱 엄격하게 (0.35 -> 0.25) 수정하여 손을 굽히는 준비 자세 오입력 완벽 차단
                pinch_th_rh = hand_scale * 0.25
                release_th_rh = hand_scale * 0.35
                
                # 포인터 시각화 (오른손 검지)
                pointer_rx, pointer_ry = int(idx_pos[0] * w_img), int(idx_pos[1] * h_img)
                
                # 거리 및 상태에 따른 다이내믹 컬러/두께 적용
                if self.current_state == HandState.DRAWING:
                    pointer_color = (0, 255, 0) # 드로잉 중 (초록색)
                    pointer_thickness = -1      # 꽉 찬 원
                else:
                    # 핀치 준비 중이면 노란색, 멀면 빨간색
                    pointer_color = (0, 255, 255) if dist_3d < release_th_rh else (0, 0, 255)
                    pointer_thickness = 2       # 빈 원
                    
                cv2.circle(display_image, (pointer_rx, pointer_ry), 6, pointer_color, pointer_thickness)
                
                z_val = np.clip((hand_scale - 0.05) / 0.2, 0, 1)
                raw_x, raw_y, raw_z = (idx_pos[0] + thumb_pos[0]) / 2, (idx_pos[1] + thumb_pos[1]) / 2, z_val

                # 주먹 상태 판별 (O자 모양 핀치를 방해하지 않으면서 느슨한 주먹만 차단하도록 1.1 -> 0.95로 조정)
                idx_folded = calc_dist_rh(8, 0) < calc_dist_rh(5, 0) * 0.95
                idx_folded_lock = calc_dist_rh(8, 0) < calc_dist_rh(5, 0) * 0.75  # 드로잉 락
                
                # 나머지 손가락(중지, 약지, 새끼)이 주먹처럼 접혀 있는지 확인 (준비 자세 오입력 완벽 차단)
                mid_folded = calc_dist_rh(12, 0) < calc_dist_rh(9, 0) * 1.1
                ring_folded = calc_dist_rh(16, 0) < calc_dist_rh(13, 0) * 1.1
                pinky_folded = calc_dist_rh(20, 0) < calc_dist_rh(17, 0) * 1.1
                other_fingers_folded = mid_folded and ring_folded and pinky_folded

                # 드로잉 의도 판별: 확실한 핀치 + 손가락 화면 이탈 환각 방지 + 나머지 손가락은 주먹 쥔 상태여야 함
                is_idx_in_bounds_rh = all((0.02 < lm[i].x < 0.98) and (0.02 < lm[i].y < 0.98) for i in [5, 6, 7, 8])
                is_pinch_intent = is_idx_in_bounds_rh and (dist_3d < pinch_th_rh) and (dist_3d < calc_dist_rh(8, 7) * 1.2) and not idx_folded and other_fingers_folded and not both_fists
                
                # 드로잉 락 (Hysteresis): 너무 안 풀리는 현상 방지 (1.5 -> 1.2)
                is_pinch = (dist_3d < release_th_rh * 1.2 and not idx_folded_lock and not both_fists) if self.current_state == HandState.DRAWING else is_pinch_intent
                
                # 오른손 마우스 포인팅 로직 (오른손 검지만 펴고 있을 때만 시스템 마우스 제어)
                idx_extended_rh = calc_dist_rh(8, 0) > calc_dist_rh(5, 0) * 1.3
                if is_idx_in_bounds_rh and not left_interacting and not is_pinch and idx_extended_rh and other_fingers_folded:
                    self.mouse.move(idx_pos[0], idx_pos[1])
                
                # --- [상태 전이 로직] ---
                if left_interacting:
                    # 왼손 동작 시 오른손은 IDLE 전환 (동시 입력 방지)
                    if self.current_state == HandState.DRAWING and len(self.active_stroke) > 0:
                        self.all_strokes.append(self.active_stroke)
                        self.active_stroke = []
                        self.undone_strokes.clear()
                    # 만약 왼손이 UI SELECT 중이면 상태를 덮어쓰지 않고 유지
                    if self.current_state != HandState.MENU_SELECTION:
                        self.current_state = HandState.IDLE
                else:
                    if self.current_state == HandState.IDLE:
                        if is_pinch:
                            self.current_state = HandState.DRAWING
                            self.pinch_lost_frames = 0
                            self.draw_state_mean = np.array([raw_x, raw_y, raw_z, 0, 0, 0])
                            self.draw_state_cov = np.eye(6)
                            self.sx, self.sy, self.sz = raw_x, raw_y, raw_z
                    elif self.current_state == HandState.DRAWING:
                        if not is_pinch:
                            self.pinch_lost_frames += 1
                            if self.pinch_lost_frames > 3:
                                if len(self.active_stroke) > 0:
                                    self.all_strokes.append(self.active_stroke)
                                    self.active_stroke = []
                                    self.undone_strokes.clear()
                                self.current_state = HandState.IDLE
                        else:
                            self.pinch_lost_frames = 0

                # --- [상태별 동작 실행] ---
                if self.current_state == HandState.IDLE:
                    pass # 평소에는 빨간/노란 연결선(cv2.line)을 그리지 않음
                    
                elif self.current_state == HandState.DRAWING:
                    self.draw_state_mean, self.draw_state_cov = self.kf_draw.filter_update(
                        self.draw_state_mean, self.draw_state_cov, np.array([raw_x, raw_y, raw_z])
                    )
                    self.sx, self.sy, self.sz = self.draw_state_mean[:3]
                    
                    cx, cy, cz = 0.5, 0.5, 0.5
                    ix, iy, iz = (self.sx - cx)/self.view_zoom, (self.sy - cy)/self.view_zoom, (self.sz - cz)/self.view_zoom
                    ax, ay = -self.view_rot_x, -self.view_rot_y
                    iy_n = iy * np.cos(ax) - iz * np.sin(ax); iz_n = iy * np.sin(ax) + iz * np.cos(ax); iy, iz = iy_n, iz_n
                    ix_n = ix * np.cos(ay) - iz * np.sin(ay); iz_n = ix * np.sin(ay) + iz * np.cos(ay); ix, iz = ix_n, iz_n
                    
                    self.active_stroke.append((ix + cx, iy + cy, iz + cz))
                    cv2.circle(display_image, (int(self.sx*w_img), int(self.sy*h_img)), 15, (0, 255, 0), -1)
                    cv2.putText(display_image, "DRAWING", (50, 100), 0, 1, (0, 255, 0), 2)
                    
                    # 드로잉 중일 때만 엄지와 검지 사이에 초록색 연결선을 그림
                    cv2.line(display_image, (int(idx_pos[0]*w_img), int(idx_pos[1]*h_img)), (int(thumb_pos[0]*w_img), int(thumb_pos[1]*h_img)), (0, 255, 0), 3)

            else:
                self.right_hand_lost_frames += 1
                if self.current_state == HandState.DRAWING and self.right_hand_lost_frames < 5:
                    pass # 빠른 이동으로 인한 일시적인 손실 시 드로잉 유지
                else:
                    # 오른손이 화면 밖으로 완전히 나갔을 때 드로잉 상태 초기화
                    if self.current_state == HandState.DRAWING:
                        if len(self.active_stroke) > 0:
                            self.all_strokes.append(self.active_stroke)
                            self.active_stroke = []
                            self.undone_strokes.clear()
                        self.current_state = HandState.IDLE

            self.draw_strokes_3d(display_image, w_img, h_img)
            self.render_minimaps(display_image, w_img, h_img)

            cv2.imshow('3D Virtual Touch Painter', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                while self.all_strokes:
                    self.undone_strokes.append(self.all_strokes.pop())
                self.active_stroke.clear()
            elif key == ord('u') or key == ord('U'):
                if self.all_strokes:
                    self.undone_strokes.append(self.all_strokes.pop())
            elif key == ord('r') or key == ord('R'):
                if self.undone_strokes:
                    self.all_strokes.append(self.undone_strokes.pop())

    def draw_strokes_3d(self, img, w, h):
        strokes_to_draw = self.all_strokes + ([self.active_stroke] if self.active_stroke else [])
        ay, ax = self.view_rot_y, self.view_rot_x
        zoom, cx, cy, cz = self.view_zoom, 0.5, 0.5, 0.5
        for stroke in strokes_to_draw:
            if len(stroke) < 2: continue
            for i in range(1, len(stroke)):
                rotated_pts = []
                for p in [stroke[i-1], stroke[i]]:
                    x, y, z = (p[0] - cx) * zoom, (p[1] - cy) * zoom, (p[2] - cz) * zoom
                    x_n = x * np.cos(ay) - z * np.sin(ay); z_n = x * np.sin(ay) + z * np.cos(ay); x, z = x_n, z_n
                    y_n = y * np.cos(ax) - z * np.sin(ax); z_n = y * np.sin(ax) + z * np.cos(ax); y, z = y_n, z_n
                    rotated_pts.append((x + cx, y + cy, z + cz))
                p1, p2 = rotated_pts[0], rotated_pts[1]
                z_clamped = np.clip(p2[2], 0, 1)
                color = (int(255 * z_clamped), 50, int(255 * (1 - z_clamped)))
                thickness = int(max(1, 2 + p2[2] * 12))
                cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, thickness)

    def render_minimaps(self, img, w, h):
        m_size = 250
        top_view = np.zeros((m_size, m_size, 3), dtype=np.uint8); side_view = np.zeros((m_size, m_size, 3), dtype=np.uint8)
        cv2.putText(top_view, "TOP (X-Z)", (10, 25), 0, 0.6, (255,255,255), 1); cv2.putText(side_view, "SIDE (Z-Y)", (10, 25), 0, 0.6, (255,255,255), 1)
        strokes_to_draw = self.all_strokes + ([self.active_stroke] if self.active_stroke else [])
        for stroke in strokes_to_draw:
            if len(stroke) < 2: continue
            for i in range(1, len(stroke)):
                p1, p2 = stroke[i-1], stroke[i]
                color = (int(255 * p2[2]), 100, int(255 * (1-p2[2])))
                cv2.line(top_view, (int(p1[0]*m_size), int((1-p1[2])*m_size)), (int(p2[0]*m_size), int((1-p2[2])*m_size)), color, 2)
                cv2.line(side_view, (int(p1[2]*m_size), int(p1[1]*m_size)), (int(p2[2]*m_size), int(p2[1]*m_size)), color, 2)
        try:
            img[20:20+m_size, w-m_size-20 : w-20] = top_view
            img[40+m_size:40+2*m_size, w-m_size-20 : w-20] = side_view
        except: pass
