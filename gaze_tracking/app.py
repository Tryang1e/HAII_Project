import cv2
import time
import numpy as np
from core.camera import get_camera
from core.kiosk import KioskUI
from detector.eyegaze.thread import EyeGazeThread
import pyautogui

class InteractionApp:
    def __init__(self):
        self.camera = get_camera(width=640, height=480, fps=60)
        
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Gaze tracker
        self.gaze_tracker = EyeGazeThread(
            server=None, display=True, use_thread=False,
            display_size=[self.screen_w, self.screen_h], auto_cali=True, sr_selection="left",
            gaze_mode="combine"
        )
        self.exit_requested = False

        # Kiosk UI test window
        self.kiosk_ui = KioskUI()

        # 시스템 전역 마우스 제어 및 체류(Dwell) 클릭 변수
        self.dwell_start = time.time()
        self.last_mx = -1
        self.last_my = -1
        self.dwell_radius = 60
        self.dwell_threshold = 1.5
        
        self.smoothed_x = -1.0
        self.smoothed_y = -1.0
        self.ema_alpha = 0.2 # 마우스 민감도 (클수록 빠르고 즉각적이나 덜덜거림 증가, 작을수록 부드럽지만 느림)

        # Button properties: Top-Right corner
        self.btn_w, self.btn_h = 160, 60
        self.btn_x = self.screen_w - self.btn_w - 20
        self.btn_y = 20

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
        first_frame = True
        
        while not self.exit_requested:
            ir_image = self.camera.get_frame()
            if ir_image is None:
                continue
                
            if first_frame:
                self.gaze_tracker.gaze_tracker.calibration()
                first_frame = False
            
            # GazeTracker expects a Grayscale frame
            if len(ir_image.shape) == 3:
                ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
                
            h_img, w_img = ir_image.shape[:2]
            
            # Gaze Tracking
            self.gaze_tracker.process(ir_image, w_img, h_img, "left")
            
            # 실제 마우스 커서 1:1 매칭 및 체류 클릭 처리
            if self.gaze_tracker.gaze_tracker.mode != "calibration" and self.gaze_tracker.gaze_x is not None and self.gaze_tracker.gaze_x > 0:
                if self.gaze_tracker.gaze_x > self.gaze_tracker.margined_left + int(0.05*self.gaze_tracker.margined_width):
                    
                    gx = self.gaze_tracker.gaze_x
                    gy = self.gaze_tracker.gaze_y
                


                    # 시선 좌표 평활화 (EMA 적용)
                    if self.smoothed_x == -1.0:
                        self.smoothed_x = gx
                        self.smoothed_y = gy
                        
                    self.smoothed_x = (self.ema_alpha * gx) + ((1.0 - self.ema_alpha) * self.smoothed_x)
                    self.smoothed_y = (self.ema_alpha * gy) + ((1.0 - self.ema_alpha) * self.smoothed_y)
                    
                    mx = int(self.smoothed_x)
                    my = int(self.smoothed_y)
                    
                    # 시스템 마우스 1:1 직접 이동
                    try:
                        pyautogui.moveTo(mx, my, _pause=False)
                    except pyautogui.FailSafeException:
                        pass
                        
                    # 마우스 Dwell 클릭 로직
                    if self.last_mx == -1:
                        self.last_mx = mx
                        self.last_my = my
                        self.dwell_start = time.time()
                    else:
                        dist = np.sqrt((mx - self.last_mx)**2 + (my - self.last_my)**2)
                        
                        target_elapsed = 0.0
                        if dist < self.dwell_radius:
                            target_elapsed = time.time() - self.dwell_start
                            if target_elapsed >= self.dwell_threshold:
                                try:
                                    pyautogui.click(x=mx, y=my, _pause=False)
                                except pyautogui.FailSafeException:
                                    pass
                                self.last_mx = -1 # 클릭 어뷰징 방지
                                self.dwell_start = time.time()
                                target_elapsed = 0.0
                        else:
                            self.last_mx = mx
                            self.last_my = my
                            self.dwell_start = time.time()
                            
                        # Kiosk UI에 시각적 피드백 (포인터 및 게이지) 전달
                        ratio = min(target_elapsed / self.dwell_threshold, 1.0)
                        self.kiosk_ui.update_indicator(mx, my, ratio)
            
            self.kiosk_ui.show()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or self.gaze_tracker.is_close or self.kiosk_ui.exit_requested: 
                break
            elif key == ord('c') or key == ord('C'):
                self.gaze_tracker.gaze_tracker.calibration()
