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
        self.left_state = HandState.IDLE
        self.right_state = HandState.IDLE
        
        self.view_rot_x = 0  
        self.view_rot_y = 0  
        self.view_zoom = 1.0
        self.prev_rot_pos = None
        self.prev_zoom_y = None
        self.is_2d_mode = False  # 2D 고정 모드 플래그
        self.current_cursor_3d = None  # 미니맵에 표시할 현재 3D 커서 위치

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
        self.rotation_lost_frames = 0
        self.zoom_lost_frames = 0
        self.smooth_lx, self.smooth_ly = None, None

        # --- [SVM Model Initialization] ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        svm_path = os.path.join(base_dir, 'virtual-touch-click', 'gesture_svm_model.xml')
        labels_path = os.path.join(base_dir, 'virtual-touch-click', 'gesture_labels.json')
        self.svm = cv2.ml.SVM_load(svm_path)
        with open(labels_path, 'r') as f:
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

    def process_frame(self, draw_ui=False):
        if True:
            frame = self.camera.get_frame()
            if frame is None: return None
            
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

            # 제스처 상태 변수 초기화 (손이 없을 때도 에러가 나지 않도록)
            is_zoom = False
            is_rotation = False
            is_ui_select = False
            is_pointer = False

            # 1. 사용자 왼손 (회전 및 줌)
            left_interacting = False
            if user_left:
                raw_lx, raw_ly = user_left['pos']
                if self.smooth_lx is None:
                    self.smooth_lx, self.smooth_ly = raw_lx, raw_ly
                else:
                    self.smooth_lx = self.smooth_lx * 0.7 + raw_lx * 0.3
                    self.smooth_ly = self.smooth_ly * 0.7 + raw_ly * 0.3
                lx, ly = self.smooth_lx, self.smooth_ly
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
                pinch_th = palm_size * 0.55      # 핀치 시작 임계값 대폭 완화
                release_th = palm_size * 0.70    # 핀치 해제 임계값 대폭 완화
                clear_th = palm_size * 0.80      # 확실히 폈다고 인정할 임계값
                
                # 왼손 검지가 펴져 있는지 확실하게 판별 (Tip-Wrist 거리가 MCP-Wrist 거리의 1.3배 이상)
                idx_extended = calc_dist(8, 0) > calc_dist(5, 0) * 1.3
                mid_extended = calc_dist(12, 0) > calc_dist(9, 0) * 1.3
                is_left_open = idx_extended and mid_extended
                
                # 손가락 접힘 판별 (완전한 꽉 쥔 주먹이 아니어도 인식되도록 1.1 -> 1.35로 매우 너그럽게 수정)
                idx_tucked = calc_dist(8, 0) < calc_dist(5, 0) * 1.35
                mid_tucked = calc_dist(12, 0) < calc_dist(9, 0) * 1.35
                ring_tucked = calc_dist(16, 0) < calc_dist(13, 0) * 1.35
                pinky_tucked = calc_dist(20, 0) < calc_dist(17, 0) * 1.35
                
                # 확실히 접힌 상태(완전한 주먹/포인팅 상태) 판별 (O모양 핀치와 구분하기 위함)
                idx_tightly_tucked = calc_dist(8, 0) < calc_dist(5, 0) * 0.95
                mid_tightly_tucked = calc_dist(12, 0) < calc_dist(9, 0) * 0.95
                
                # 주먹(Rotation) 판별: 새끼나 약지가 덜 접혀도 검지/중지만 확실히 접히면 인식되도록 완화
                is_left_fist = idx_tucked and mid_tucked
                both_fists = is_left_fist and is_right_fist
                
                # 주먹 잠금용 (손목 각도가 틀어져도 풀리지 않도록 매우 넉넉한 임계값 적용: 1.3 -> 1.6)
                idx_lock_tucked = calc_dist(8, 0) < calc_dist(5, 0) * 1.6
                mid_lock_tucked = calc_dist(12, 0) < calc_dist(9, 0) * 1.6

                # 양손이 모두 주먹일 때는 모든 동작 무시
                can_left_interact = not both_fists

                # 손가락 관절 중 하나라도 카메라 시야의 극단적 가장자리(2% 이내)에 있으면 
                # 화면 밖으로 나간 것으로 간주하여 MediaPipe의 환각(Hallucination) 핀치를 차단
                is_idx_in_bounds = all((0.02 < lm[i].x < 0.98) and (0.02 < lm[i].y < 0.98) for i in [5, 6, 7, 8])
                is_mid_in_bounds = all((0.02 < lm[i].x < 0.98) and (0.02 < lm[i].y < 0.98) for i in [9, 10, 11, 12])

                # --- [동작 상태 잠금 (Hysteresis Lock) 및 우선순위 결정] ---
                
                # [상호 배타적 제스처 판별 로직]
                # 각 기능이 겹치지 않도록 검지와 중지의 '펴짐(Extended)' 상태를 기준으로 역할을 완벽히 분리합니다.
                idx_ratio = calc_dist(8, 0) / calc_dist(5, 0)
                mid_ratio = calc_dist(12, 0) / calc_dist(9, 0)
                
                idx_ext = idx_ratio > 1.2   # 검지 펴짐
                mid_ext = mid_ratio > 1.2   # 중지 펴짐
                idx_tight = idx_ratio < 0.95 # 검지 꽉 접힘
                mid_tight = mid_ratio < 0.95 # 중지 꽉 접힘

                # 1. ROTATION (주먹): 검지와 중지가 모두 접힘 (가장 직관적인 주먹 형태)
                is_rotation_intent = can_left_interact and not is_right_fist and (not idx_ext) and (not mid_ext)
                
                # 2. CLICK (검지 핀치): 검지-엄지 핀치 (중지는 반드시 펴져 있어야 함 -> 주먹이나 Zoom과 절대 안 겹침)
                is_ui_select_intent = can_left_interact and is_idx_in_bounds and (dist_thumb_idx < pinch_th) and mid_ext and not idx_tight
                
                # 3. ZOOM (중지 핀치): 중지-엄지 핀치 (검지는 반드시 펴져 있어야 함 -> 주먹이나 Click과 절대 안 겹침)
                is_zoom_intent = can_left_interact and is_mid_in_bounds and (dist_thumb_mid < pinch_th) and idx_ext and not mid_tight
                
                # 4. POINTER (포인터): 검지만 펴지고 중지는 접힘 (Zoom 핀치가 아닐 때만 발동)
                is_pointer_intent = can_left_interact and is_idx_in_bounds and idx_ext and (not mid_ext) and not is_zoom_intent
                
                # 1순위: Rotation 유지 및 진입
                if is_rotation_intent or (self.prev_rot_pos is not None and (not idx_ext) and (not mid_ext) and not is_right_fist):
                    is_rotation = True
                # 2순위: Zoom 유지 (해제 임계값까지)
                elif self.prev_zoom_y is not None and dist_thumb_mid < release_th:
                    is_zoom = True
                # 3순위: Click 유지 (해제 임계값까지)
                elif self.left_state == HandState.MENU_SELECTION and dist_thumb_idx < release_th:
                    is_ui_select = True
                # 4순위: 현재 프레임의 의도 반영
                else:
                    is_zoom = is_zoom_intent
                    is_ui_select = is_ui_select_intent
                    
                # 2D 고정 모드일 때는 회전 기능만 차단 (줌은 유지/사용 가능)
                if self.is_2d_mode:
                    is_rotation = False
                    self.prev_rot_pos = None
                    
                if not is_zoom and not is_rotation and not is_ui_select:
                    is_pointer = is_pointer_intent
                    
                # 동작 락 보정 (프레임 드랍 대응)
                if is_rotation:
                    self.rotation_lost_frames = 0
                elif self.prev_rot_pos is not None:
                    self.rotation_lost_frames += 1
                    if self.rotation_lost_frames < 4:
                        is_rotation = True

                if is_zoom:
                    self.zoom_lost_frames = 0
                elif self.prev_zoom_y is not None:
                    self.zoom_lost_frames += 1
                    if self.zoom_lost_frames < 4:
                        is_zoom = True

                # 동작 종료 시 상태 변수 초기화
                if not is_zoom: self.prev_zoom_y = None
                if not is_rotation: self.prev_rot_pos = None
                if not is_ui_select and self.left_state == HandState.MENU_SELECTION:
                    self.left_state = HandState.IDLE
                
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
                    self.left_state = HandState.MENU_SELECTION
                    
                    if is_pointer:
                        self.mouse.move(lx, ly)
                        cv2.circle(display_image, (pointer_lx, pointer_ly), 6, (255, 0, 255), -1)
                    
                    if is_ui_select:
                        cv2.putText(display_image, "UI SELECT (CLICK)", (50, 150), 0, 1, (255, 0, 255), 2)
                        cv2.circle(display_image, (pointer_lx, pointer_ly), 8, (0, 0, 255), -1) # 클릭 시 빨간색
                        self.mouse.click()
                else:
                    if self.left_state == HandState.MENU_SELECTION:
                        self.left_state = HandState.IDLE
                    self.prev_rot_pos, self.prev_zoom_y = None, None
            else:
                # 왼손이 화면 밖으로 완전히 나갔을 때 상태 초기화
                if self.left_state == HandState.MENU_SELECTION:
                    self.left_state = HandState.IDLE
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
                # 드로잉 의도 임계값을 부드럽게 조정 (0.25 -> 0.30)
                pinch_th_rh = hand_scale * 0.30
                release_th_rh = hand_scale * 0.45
                
                # 포인터 시각화 (오른손 검지)
                pointer_rx, pointer_ry = int(idx_pos[0] * w_img), int(idx_pos[1] * h_img)
                
                # 거리 및 상태에 따른 다이내믹 컬러/두께 적용
                if self.right_state == HandState.DRAWING:
                    pointer_color = (0, 255, 0) # 드로잉 중 (초록색)
                    pointer_thickness = -1      # 꽉 찬 원
                else:
                    # 핀치 준비 중이면 노란색, 멀면 빨간색
                    pointer_color = (0, 255, 255) if dist_3d < release_th_rh else (0, 0, 255)
                    pointer_thickness = 2       # 빈 원
                    
                cv2.circle(display_image, (pointer_rx, pointer_ry), 6, pointer_color, pointer_thickness)
                
                z_val = np.clip((hand_scale - 0.05) / 0.2, 0, 1)
                if self.is_2d_mode:
                    z_val = 0.5  # 2D 모드일 때는 깊이를 화면 중앙(0.5)으로 완전 고정
                raw_x, raw_y, raw_z = (idx_pos[0] + thumb_pos[0]) / 2, (idx_pos[1] + thumb_pos[1]) / 2, z_val
                
                self.current_cursor_3d = (raw_x, raw_y, raw_z)

                # 주먹 상태 판별 (O자 모양 핀치를 방해하지 않으면서 느슨한 주먹만 차단하도록 1.1 -> 0.95로 조정)
                idx_folded = calc_dist_rh(8, 0) < calc_dist_rh(5, 0) * 0.95
                idx_folded_lock = calc_dist_rh(8, 0) < calc_dist_rh(5, 0) * 0.75  # 드로잉 락
                
                # 나머지 손가락(중지, 약지, 새끼)이 주먹처럼 접혀 있는지 확인 (준비 자세 오입력 완벽 차단)
                mid_folded = calc_dist_rh(12, 0) < calc_dist_rh(9, 0) * 1.1
                ring_folded = calc_dist_rh(16, 0) < calc_dist_rh(13, 0) * 1.1
                pinky_folded = calc_dist_rh(20, 0) < calc_dist_rh(17, 0) * 1.1
                other_fingers_folded = mid_folded and ring_folded and pinky_folded

                # 드로잉 의도 판별: 확실한 핀치 + 손가락 화면 이탈 환각 방지 + 나머지 손가락은 주먹 쥔 상태여야 함 (조건 완화: 다른 손가락 접힘 검사 제거)
                is_idx_in_bounds_rh = all((0.02 < lm[i].x < 0.98) and (0.02 < lm[i].y < 0.98) for i in [5, 6, 7, 8])
                is_pinch_intent = is_idx_in_bounds_rh and (dist_3d < pinch_th_rh) and (dist_3d < calc_dist_rh(8, 7) * 1.5) and not idx_folded and not both_fists
                
                # 드로잉 락 (Hysteresis): 너무 안 풀리는 현상 방지
                is_pinch = (dist_3d < release_th_rh * 1.2 and not idx_folded_lock and not both_fists) if self.right_state == HandState.DRAWING else is_pinch_intent
                
                # 오른손 마우스 포인팅 로직 (카메라 창 내부에서만 제어)
                idx_extended_rh = calc_dist_rh(8, 0) > calc_dist_rh(5, 0) * 1.3
                if is_idx_in_bounds_rh and not is_pinch and idx_extended_rh and other_fingers_folded:
                    self.mouse.move(idx_pos[0], idx_pos[1])
                
                # --- [상태 전이 로직 (양손 동시 상호작용 허용)] ---
                if self.right_state == HandState.IDLE:
                    if is_pinch:
                        self.right_state = HandState.DRAWING
                        self.pinch_lost_frames = 0
                        self.draw_state_mean = np.array([raw_x, raw_y, raw_z, 0, 0, 0])
                        self.draw_state_cov = np.eye(6)
                        self.sx, self.sy, self.sz = raw_x, raw_y, raw_z
                elif self.right_state == HandState.DRAWING:
                    if not is_pinch:
                        self.pinch_lost_frames += 1
                        if self.pinch_lost_frames > 3:
                            if len(self.active_stroke) > 0:
                                self.all_strokes.append(self.active_stroke)
                                self.active_stroke = []
                                self.undone_strokes.clear()
                            self.right_state = HandState.IDLE
                    else:
                        self.pinch_lost_frames = 0

                # --- [상태별 동작 실행] ---
                if self.right_state == HandState.IDLE:
                    pass # 평소에는 빨간/노란 연결선(cv2.line)을 그리지 않음
                    
                elif self.right_state == HandState.DRAWING:
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
                if self.right_state == HandState.DRAWING and self.right_hand_lost_frames < 5:
                    pass # 빠른 이동으로 인한 일시적인 손실 시 드로잉 유지
                else:
                    # 오른손이 화면 밖으로 완전히 나갔을 때 드로잉 상태 초기화
                    if self.right_state == HandState.DRAWING:
                        if len(self.active_stroke) > 0:
                            self.all_strokes.append(self.active_stroke)
                            self.active_stroke = []
                            self.undone_strokes.clear()
                        self.right_state = HandState.IDLE
                
                self.current_cursor_3d = None

            self.draw_strokes_3d(display_image, w_img, h_img)
            self.render_minimaps(display_image, w_img, h_img, draw_ui)
            
            # GUI 앱에서 상태를 읽어갈 수 있도록 인스턴스 변수에 저장
            self.is_rotation = is_rotation
            self.is_zoom = is_zoom
            self.is_ui_select = is_ui_select
            
            if draw_ui:
                self.draw_premium_ui(display_image, w_img, h_img, is_rotation, is_zoom, is_ui_select)
            
            return display_image

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

    def render_minimaps(self, img, w, h, draw_ui=True):
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
        
        # 현재 사용자의 손 위치(커서)를 미니맵에 표시
        if self.current_cursor_3d is not None:
            cx, cy, cz = self.current_cursor_3d
            
            # 화면(Screen) 좌표를 현재 카메라 뷰(Zoom, Rotation)를 반영하여 실제 3D 월드(World) 좌표로 변환
            ix, iy, iz = (cx - 0.5)/self.view_zoom, (cy - 0.5)/self.view_zoom, (cz - 0.5)/self.view_zoom
            ax, ay = -self.view_rot_x, -self.view_rot_y
            iy_n = iy * np.cos(ax) - iz * np.sin(ax); iz_n = iy * np.sin(ax) + iz * np.cos(ax); iy, iz = iy_n, iz_n
            ix_n = ix * np.cos(ay) - iz * np.sin(ay); iz_n = ix * np.sin(ay) + iz * np.cos(ay); ix, iz = ix_n, iz_n
            
            world_x, world_y, world_z = ix + 0.5, iy + 0.5, iz + 0.5
            
            # 깜빡임 효과 (애니메이션)
            pulse = int(5 + 2 * np.sin(time.time() * 10))
            
            # TOP (X-Z) map: x는 X축, y는 Z축
            top_px, top_py = int(world_x * m_size), int((1 - world_z) * m_size)
            cv2.circle(top_view, (top_px, top_py), pulse, (0, 0, 255), -1)
            cv2.circle(top_view, (top_px, top_py), pulse + 3, (0, 0, 255), 1)
            cv2.putText(top_view, "YOU", (top_px + 10, top_py), 0, 0.4, (0, 0, 255), 1)
            
            # SIDE (Z-Y) map: x는 Z축, y는 Y축
            side_px, side_py = int(world_z * m_size), int(world_y * m_size)
            cv2.circle(side_view, (side_px, side_py), pulse, (0, 0, 255), -1)
            cv2.circle(side_view, (side_px, side_py), pulse + 3, (0, 0, 255), 1)
            cv2.putText(side_view, "YOU", (side_px + 10, side_py), 0, 0.4, (0, 0, 255), 1)
            
        self.top_view_img = top_view
        self.side_view_img = side_view
            
        if draw_ui:
            try:
                img[20:20+m_size, w-m_size-20 : w-20] = top_view
                img[40+m_size:40+2*m_size, w-m_size-20 : w-20] = side_view
            except: pass

    def draw_premium_ui(self, img, w, h, is_rotation, is_zoom, is_ui_select):
        overlay = img.copy()
        
        # Colors (BGR)
        bg_panel = (30, 30, 30)
        border_color = (80, 80, 80)
        text_primary = (245, 245, 245)
        text_secondary = (180, 180, 180)
        accent = (255, 160, 0) # Light blue/Cyan-ish or Orange. OpenCV is BGR. Let's use Cyan: (255, 200, 0)
        accent = (255, 180, 50)
        
        # 1. Top Toolbar (Tools & Actions)
        # Main Bar
        cv2.rectangle(overlay, (w//2 - 250, 15), (w//2 + 250, 65), bg_panel, -1)
        cv2.rectangle(overlay, (w//2 - 250, 15), (w//2 + 250, 65), border_color, 1)
        
        # 2. Left Sidebar (Modes)
        cv2.rectangle(overlay, (15, 80), (160, 240), bg_panel, -1)
        cv2.rectangle(overlay, (15, 80), (160, 240), border_color, 1)
        
        # 3. Status Board (Bottom)
        cv2.rectangle(overlay, (w//2 - 150, h - 70), (w//2 + 150, h - 15), bg_panel, -1)
        cv2.rectangle(overlay, (w//2 - 150, h - 70), (w//2 + 150, h - 15), border_color, 1)

        # Alpha blending for glassmorphism
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # --- Draw Texts/Icons on `img` (Opaque) ---
        
        # Top Toolbar Texts
        # Tools
        cv2.putText(img, "BRUSH", (w//2 - 220, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, accent, 1)
        cv2.putText(img, "ERASER", (w//2 - 130, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_secondary, 1)
        
        # Separator
        cv2.line(img, (w//2 - 40, 25), (w//2 - 40, 55), border_color, 1)
        
        # Actions
        cv2.putText(img, "UNDO (U)", (w//2 - 10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_primary, 1)
        cv2.putText(img, "REDO (R)", (w//2 + 80, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_primary, 1)
        cv2.putText(img, "CLEAR (C)", (w//2 + 170, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)

        # Left Sidebar Texts
        cv2.putText(img, "VIEW MODE", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_secondary, 1)
        if self.is_2d_mode:
            cv2.putText(img, "2D Plane", (30, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, accent, 1)
        else:
            cv2.putText(img, "3D Space", (30, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, accent, 1)
            
        cv2.putText(img, "Press 'P'", (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_secondary, 1)
        cv2.putText(img, "to toggle", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_secondary, 1)

        # Status Board Texts
        status_text = "IDLE"
        status_color = text_secondary
        
        if self.right_state == HandState.DRAWING:
            status_text = "DRAWING"
            status_color = (50, 255, 50) # Green
        elif is_rotation:
            status_text = "ROTATING"
            status_color = (255, 255, 50) # Cyan
        elif is_zoom:
            status_text = "ZOOMING"
            status_color = (50, 255, 255) # Yellow
        elif is_ui_select:
            status_text = "UI SELECT"
            status_color = (255, 50, 255) # Magenta
            
        cv2.putText(img, "SYSTEM STATUS", (w//2 - 120, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_secondary, 1)
        cv2.putText(img, status_text, (w//2 - 120, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

