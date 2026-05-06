import cv2
import time
import numpy as np
from core.camera import get_camera
from core.mouse import VirtualMouse
from core.ui import UIManager
from core.kiosk import KioskUI
from detector.landmark import Detector

class InteractionApp:
    def __init__(self):
        self.camera = get_camera(width=1280, height=720, fps=30)
        
        # 가상 마우스 ROI
        self.roi_x_min = 0.2
        self.roi_x_max = 0.8
        self.roi_y_min = 0.2
        self.roi_y_max = 0.8
        
        self.mouse = VirtualMouse(smoothing=0.5, 
                                  roi_x_min=self.roi_x_min, roi_x_max=self.roi_x_max,
                                  roi_y_min=self.roi_y_min, roi_y_max=self.roi_y_max)
        self.detector = Detector()
        self.ui = UIManager()
        self.kiosk = KioskUI()
        
        self.current_mode = 'hand_tracking'
        
        # --- 모드 전환 구역 설정 (우측 하단) ---
        self.zone_x_min = 0.85
        self.zone_y_min = 0.75
        self.zone_y_max = 0.95
        
        self.gesture_timer_start = None
        self.mode_switched_this_session = False
        self.time_to_mode_switch = 3.0
        
        # --- [수정됨] 영역 진입 시 오작동 방지를 위한 쿨다운 로직 변수 ---
        self.hand_visible_since = None
        self.hand_visibility_threshold = 0.5  # 손이 화면에 0.5초 이상 연속 인식되어야 클릭 허용

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
            #이미지열기
            ir_image = self.camera.get_frame()
            if ir_image is None:
                continue
                
            if len(ir_image.shape) == 2 or (len(ir_image.shape) == 3 and ir_image.shape[2] == 1):
                display_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = ir_image.copy()
            h_img, w_img = display_image.shape[:2]

            # 핸드 트래킹 실행
            self.detector.process_hands(display_image)
            
            # 랜드마크 항상 표시
            self.ui.draw_landmarks(display_image, self.detector, w_img, h_img)

            # 사람기준 왼손 중심 위치를 우선으로 하고, 없으면 오른손 중심 위치 사용
            is_left_hand_active = True
            hand_pos = self.detector.get_left_hand_pos()
            if not hand_pos:
                hand_pos = self.detector.get_right_hand_pos()
                is_left_hand_active = False
            #hand_pos = self.detector.get_right_hand_pos()


            # --- 구역 시각화 (UI) ---
            zx_sx = int(self.zone_x_min * w_img)
            zx_ex = w_img - 10
            zy_sy = int(self.zone_y_min * h_img)
            zy_ey = int(self.zone_y_max * h_img)
            
            # 구역 외곽선
            self.ui.draw_zone(display_image, zx_sx, zy_sy, zx_ex, zy_ey)

            in_zone = False
            if hand_pos:
                # --- [수정됨] 처음 손이 인식된(들어온) 시간 기록 ---
                if self.hand_visible_since is None:
                    self.hand_visible_since = time.time()
                    
                hx, hy = hand_pos
                
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
                    
                    # 두 손가락 (현재 활성화된 손의 엄지와 검지) 클릭 감지 로직
                    thumb_pos = self.detector.get_thumb_pos(is_left=is_left_hand_active)
                    index_pos = self.detector.get_index_pos(is_left=is_left_hand_active)
                    
                    if thumb_pos and index_pos:
                        # 유클리디안 거리(Euclidean Distance) 계산
                        dist = ((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)**0.5
                        
                        try:
                            pt1 = (int(thumb_pos[0] * w_img), int(thumb_pos[1] * h_img))
                            pt2 = (int(index_pos[0] * w_img), int(index_pos[1] * h_img))
                            
                            # 거리가 가까울수록 두께가 두꺼워지도록 설정 (최대 두께 15, 최소 두께 1)
                            thickness = max(1, int(15 - dist * 100))
                            cv2.line(display_image, pt1, pt2, (0, 0, 255), thickness)
                        except Exception as e:
                            print(f"Error drawing line between fingers: {e}")

                        # --- [수정됨] 클릭 조건 변경 ---
                        # 손가락 사이 거리(dist)가 0.03 미만이면서 동시에 화면에 손이 들어온 지 threshold(0.5초) 이상 지났을 때만 클릭
                        if dist < 0.05 and (time.time() - self.hand_visible_since > self.hand_visibility_threshold):
                            self.mouse.click()
#                            print('segmentation')
                            # 클릭 시각적 피드백 (엄지와 검지 사이 중앙)
                            cx = int((thumb_pos[0] + index_pos[0]) / 2 * w_img)
                            cy = int((thumb_pos[1] + index_pos[1]) / 2 * h_img)
                            cv2.circle(display_image, (cx, cy), 15, (0, 0, 255), -1)
                            cv2.putText(display_image, "CLICK", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.ui.draw_hand_tracking_mode(display_image, w_img)
            else:
                # --- [수정됨] 손이 화면 밖으로 나가면 기록하던 진입 시간 초기화 ---
                self.hand_visible_since = None
                
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

            # 최종 출력
            output_image = cv2.resize(display_image, (w_img // 2, h_img // 2))
            cv2.imshow('Hand Tracking Project', output_image)
            cv2.setWindowTitle('Hand Tracking Project', win_title)
            
            # 독립된 키오스크 윈도우 그리기
            self.kiosk.show()
            if self.kiosk.exit_requested:
                print("Exit button pressed in Kiosk. Shutting down...")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): self.current_mode = 'virtual touch'
            elif key == ord('h'): self.current_mode = 'hand_tracking'
