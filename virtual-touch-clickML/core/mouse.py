import numpy as np
import time
from pykalman import KalmanFilter

class VirtualMouse:
    def __init__(self, roi_x_min=0.2, roi_x_max=0.8, roi_y_min=0.2, roi_y_max=0.8):
        self.screen_w, self.screen_h = 1920, 1080 # Default, updated by set_window_rect
        self.prev_mx = self.screen_w // 2
        self.prev_my = self.screen_h // 2
        self.win_rect = None
        
        # Kalman Filter 초기화
        self.kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.eye(2),
            initial_state_mean=np.array([self.prev_mx, self.prev_my]),
            initial_state_covariance=np.eye(2),
            observation_covariance=np.eye(2) * 10.0,  # 노이즈 신뢰도 (값을 낮춰 반응성/인식률 극대화)
            transition_covariance=np.eye(2) * 1.0     # 상태 변화 신뢰도
        )
        self.state_mean = np.array([self.prev_mx, self.prev_my])
        self.state_cov = np.eye(2)
        
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.roi_y_min = roi_y_min
        self.roi_y_max = roi_y_max
        
        self.last_click_time = 0
        self.click_event_triggered = False

    def set_window_rect(self, wx, wy, ww, wh):
        self.win_rect = (wx, wy, ww, wh)
        self.screen_w = ww
        self.screen_h = wh

    def move(self, hx, hy):
        tx = ( hx - self.roi_x_min ) / (self.roi_x_max - self.roi_x_min)
        ty = ( hy - self.roi_y_min ) / (self.roi_y_max - self.roi_y_min)
        tx, ty = np.clip(tx, 0, 1), np.clip(ty, 0, 1)
        
        if self.win_rect:
            wx, wy, ww, wh = self.win_rect
            # Allow mapping exactly to the program window bounds
            raw_x = wx + tx * ww
            raw_y = wy + ty * wh
        else:
            raw_x = tx * self.screen_w
            raw_y = ty * self.screen_h
        
        # Kalman filter_update 적용 
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, np.array([raw_x, raw_y])
        )
        
        mx = int(self.state_mean[0])
        my = int(self.state_mean[1])
        
        # 클리핑 (모니터 해상도 밖으로 나가지 않게)
        if self.win_rect:
            mx = max(self.win_rect[0], min(self.win_rect[0] + self.win_rect[2] - 2, mx))
            my = max(self.win_rect[1], min(self.win_rect[1] + self.win_rect[3] - 2, my))
        else:
            mx = max(0, min(self.screen_w - 2, mx))
            my = max(0, min(self.screen_h - 2, my))
        
        self.prev_mx, self.prev_my = mx, my

    def move_in_window(self, hx, hy, win_rect):
        wx, wy, ww, wh = win_rect
        # 0~1 좌표계를 카메라 윈도우 내부의 절대 픽셀 좌표로 변환
        raw_x = wx + hx * ww
        raw_y = wy + hy * wh
        
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, np.array([raw_x, raw_y])
        )
        
        mx = int(self.state_mean[0])
        my = int(self.state_mean[1])
        
        # 화면 밖으로 넘어가지 않도록 방어 코드 (PyAutoGUI 튕김 방지)
        mx = max(0, min(self.screen_w - 2, mx))
        my = max(0, min(self.screen_h - 2, my))
        
        self.prev_mx, self.prev_my = mx, my

    def click(self):
        current_time = time.time()
        # 0.5초 디바운스
        if current_time - self.last_click_time > 0.5:
            # win_rect 밖이면 클릭 무시 (프로그램 창 밖 클릭 방지)
            if self.win_rect:
                wx, wy, ww, wh = self.win_rect
                mx, my = self.prev_mx, self.prev_my
                if not (wx <= mx <= wx + ww and wy <= my <= wy + wh):
                    return # 프로그램 창 밖이면 무시
            
            self.click_event_triggered = True
            self.last_click_time = current_time
