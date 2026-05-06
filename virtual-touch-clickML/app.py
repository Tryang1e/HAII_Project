import cv2
import time
import numpy as np
import json
from collections import deque
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
        
        # 제스처 인식 관련 초기화
        try:
            self.gesture_model = cv2.ml.SVM_load('./virtual-touch-click/gesture_svm_model.xml')
            with open('./virtual-touch-click/gesture_labels.json', 'r') as f:
                self.label_map = json.load(f)
        except Exception as e:
            print(f"Failed to load gesture model: {e}")
            self.gesture_model = None
            self.label_map = {}
            
        self.gesture_buffer = deque(maxlen=20)
        self.current_gesture = "None"
        self.last_gesture_text = ""
        self.gesture_display_time = 0
        
        # 제스처 -> 키오스크 버튼 맵핑
        self.gesture_to_btn = {
            "one": "1", "two": "2", "three": "3", "four": "4", 
            "quit": "Exit", "payment": "Payment", "ok": "Home", "normal": "Home"
        }

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

            # 제스처 추론
            angles = self.detector.get_joint_angles()
            if angles is not None:
                self.gesture_buffer.append(angles)
                if len(self.gesture_buffer) == 20 and self.gesture_model is not None:
                    # 버퍼의 프레임 데이터를 1차원 배열로 평탄화 (20 frames * 15 angles = 300)
                    features = np.array(self.gesture_buffer, dtype=np.float32).flatten()
                    features = features.reshape(1, -1)
                    
                    _, result = self.gesture_model.predict(features)
                    pred_class = int(result[0][0])
                    pred_label = self.label_map.get(str(pred_class), "Unknown")
                    
                    # OpenCV SVM의 predict()는 확률을 바로 반환하지 않아, 단순 라벨만 표시
                    self.current_gesture = pred_label
                    
                    # 키오스크 호버 업데이트
                    target_btn = self.gesture_to_btn.get(self.current_gesture.lower(), None)
                    self.kiosk.set_hover(target_btn)
            else:
                self.gesture_buffer.clear()
                self.current_gesture = "None"
                self.kiosk.set_hover(None)
            
            # 1초간 UI 팝업 유지 로직 ("normal" 제스처는 알림 제외)
            if self.current_gesture != "None" and self.current_gesture.lower() != "normal":
                self.last_gesture_text = self.current_gesture
                self.gesture_display_time = time.time()

            # 인식된 제스처 화면 팝업 UI (1초 유지, 보색 대비)
            if time.time() - self.gesture_display_time < 1.0 and self.last_gesture_text:
                gesture_text = f" {self.last_gesture_text.upper()} "
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1.8
                thickness = 3
                
                (tw, th), baseline = cv2.getTextSize(gesture_text, font, font_scale, thickness)
                
                pad = 20
                
                # 좌상단 여백 20픽셀
                rx1, ry1 = 20, 20
                rx2, ry2 = rx1 + tw + 2 * pad, ry1 + th + baseline + 2 * pad
                
                bg_color = (40, 40, 40)       
                border_color = (255, 170, 0) 
                text_color = (255, 255, 255)  
                
                cv2.rectangle(display_image, (rx1, ry1), (rx2, ry2), bg_color, -1)
                cv2.rectangle(display_image, (rx1, ry1), (rx2, ry2), border_color, 2) 
                
                text_x = rx1 + pad
                text_y = ry1 + pad + th
                cv2.putText(display_image, gesture_text, (text_x, text_y+1), font, font_scale, text_color, thickness)


            # 사람기준 왼손 오른손 중심 위치 (모든 관절의 무게중심 사용)
            hand_pos = self.detector.get_left_hand_pos()

            # --- 구역 시각화 (UI) ---
            zx_sx = int(self.zone_x_min * w_img)
            zx_ex = w_img - 10
            zy_sy = int(self.zone_y_min * h_img)
            zy_ey = int(self.zone_y_max * h_img)
            
            # 구역 외곽선
            self.ui.draw_zone(display_image, zx_sx, zy_sy, zx_ex, zy_ey)

            in_zone = False
            if hand_pos:
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
                    
                    # 두 손가락 (왼손, 오른손 검지) 클릭 감지 로직
                    l_idx = self.detector.get_left_index_pos()
                    r_idx = self.detector.get_left_thumb_pos()
                    
                    if l_idx and r_idx:
                        # 유클리디안 거리(Euclidean Distance) 계산
                        dist = ((l_idx[0] - r_idx[0])**2 + (l_idx[1] - r_idx[1])**2)**0.5
                        
                        try:
                            pt1 = (int(l_idx[0] * w_img), int(l_idx[1] * h_img))
                            pt2 = (int(r_idx[0] * w_img), int(r_idx[1] * h_img))
                            
                            # 거리가 가까울수록 두께가 두꺼워지도록 설정 (최대 두께 15, 최소 두께 1)
                            thickness = max(1, int(15 - dist * 100))
                            cv2.line(display_image, pt1, pt2, (0, 0, 255), thickness)
                        except Exception as e:
                            print(f"Error drawing line between fingers: {e}")

                        if dist < 0.05: # 클릭 임계값
                            self.mouse.click()
                            
                            # Kiosk 버튼이 호버 중이라면 Kiosk 클릭도 실행
                            if self.kiosk.hovered_button:
                                self.kiosk.trigger_click()
                            
                            # 클릭 시각적 피드백 (왼손 검지 위치 기준 중앙)
                            cx = int((l_idx[0] + r_idx[0]) / 2 * w_img)
                            cy = int((l_idx[1] + r_idx[1]) / 2 * h_img)
                            cv2.circle(display_image, (cx, cy), 15, (0, 0, 255), -1)
                            cv2.putText(display_image, "CLICK", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
