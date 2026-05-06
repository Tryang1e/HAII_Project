import cv2
import time
import numpy as np
from core.camera import get_camera
from core.mouse import VirtualMouse
from core.ui import UIManager
from detector.landmark import Detector
from Sensor import GestureSensor # 수정됨: 제스처 센서 임포트

class InteractionApp:
    def __init__(self):
        self.camera = get_camera(width=1280, height=720, fps=30)
        
        # 가상 마우스 ROI
        self.roi_x_min = 0.3
        self.roi_x_max = 0.7
        self.roi_y_min = 0.3
        self.roi_y_max = 0.7
         


        self.mouse = VirtualMouse(smoothing=0.5, 
                                  roi_x_min=self.roi_x_min, roi_x_max=self.roi_x_max,
                                  roi_y_min=self.roi_y_min, roi_y_max=self.roi_y_max)
        self.detector = Detector()
        self.ui = UIManager()
        self.sensor = GestureSensor() # 수정됨: 제스처 센서 초기화
        
        self.current_mode = 'hand_tracking'
        
        # --- 모드 전환 구역 설정 (우측 하단) ---
        self.zone_x_min = 0.85
        self.zone_y_min = 0.75
        self.zone_y_max = 0.95
        
        self.gesture_timer_start = None
        self.mode_switched_this_session = False
        self.time_to_mode_switch = 3.0

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
            if frame is None:
                continue
                
            display_image = frame.copy()
            h_img, w_img = display_image.shape[:2]

            # 핸드 트래킹 실행
            self.detector.process_hands(display_image)
            
            # 랜드마크 항상 표시
            self.ui.draw_landmarks(display_image, self.detector, w_img, h_img)

            # 사람기준 왼손 오른손 중심 위치 (모든 관절의 무게중심 사용)
            hand_pos_left = self.detector.get_left_hand_pos()
            hand_pos_right = self.detector.get_right_hand_pos()
            
            hand_pos = [ hand_pos_left, hand_pos_right ]
            if hand_pos_left == None and hand_pos_right == None:
                hand_pos = None


            # --- 구역 시각화 (UI) ---
            zx_sx = int(self.zone_x_min * w_img)
            zx_ex = w_img - 10
            zy_sy = int(self.zone_y_min * h_img)
            zy_ey = int(self.zone_y_max * h_img)
            
            # 구역 외곽선
            self.ui.draw_zone(display_image, zx_sx, zy_sy, zx_ex, zy_ey)

            in_zone = False
            if hand_pos:
                if hand_pos[0]:
                    hx, hy, hand_info = hand_pos[0]
                if hand_pos[1]:
                    hx, hy, hand_info = hand_pos[1]
                
                # --- 구역 내 손 위치 체크 ---
                if hx > self.zone_x_min and hy > self.zone_y_min and hy < self.zone_y_max:
                    in_zone = True
                    if self.gesture_timer_start is None: 
                        self.gesture_timer_start = time.time()
                        self.mode_switched_this_session = False
                    
                    elapsed = time.time() - self.gesture_timer_start
                    
                    # 시각적 진행바
                    progress_h = int(min(elapsed / self.time_to_mode_switch, 1.0) * (zy_ey - zy_sy))
                    self.ui.draw_progress(display_image, zx_sx, zy_sy, zx_ex, zy_ey, progress_h)
                    
                    if elapsed >= self.time_to_mode_switch and not self.mode_switched_this_session:
                        self.current_mode = 'virtual touch' if self.current_mode == 'hand_tracking' else 'hand_tracking'
                        print(f"Mode switched to {self.current_mode}")
                        self.mode_switched_this_session = True
                else:
                    self.gesture_timer_start = None
                    self.mode_switched_this_session = False

                self.ui.draw_hand_position(display_image, hx, hy, w_img, h_img)

                # --- 마우스 제어 로직 ---
                if self.current_mode == 'virtual touch':
                    self.mouse.move(hx, hy)
                    self.ui.draw_virtual_touch_mode(display_image, self.roi_x_min, self.roi_y_min, self.roi_x_max, self.roi_y_max, w_img, h_img)
                else:
                    self.ui.draw_hand_tracking_mode(display_image, w_img)
            else:
                self.gesture_timer_start = None
                self.mode_switched_this_session = False
                
                if self.current_mode == 'virtual touch':
                    self.ui.draw_virtual_touch_mode(display_image, self.roi_x_min, self.roi_y_min, self.roi_x_max, self.roi_y_max, w_img, h_img)
                else:
                    self.ui.draw_hand_tracking_mode(display_image, w_img)

            # 윈도우 제목 상태 표시
            mode_label = self.current_mode.replace("virtual touch", "V-TOUCH").upper()
            win_title = f"Hand Tracking (Mode: {mode_label}) | Mode Switch "
            if in_zone: 
                win_title += " [ACTIVE]"

            # 수정됨: 제스처 감지 및 화면 표시 로직 ---
            # 카메라 기준 Right(사람 기준 왼손)의 랜드마크를 가져와서 제스처 판별
            left_hand_landmarks = self.detector.get_hand_landmarks(label="Right")
            detected_gesture = self.sensor.detect_gesture(left_hand_landmarks)
            
            if detected_gesture and detected_gesture != "Unknown":
                cv2.putText(display_image, f"Gesture: {detected_gesture}", (10, h_img - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # --- 수정 끝 ---

            # 최종 출력
            output_image = cv2.resize(display_image, (w_img // 2, h_img // 2))
            cv2.imshow('Hand Tracking Project', output_image)
            cv2.setWindowTitle('Hand Tracking Project', win_title)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): self.current_mode = 'virtual touch'
            elif key == ord('h'): self.current_mode = 'hand_tracking'
