import pyautogui
import numpy as np
import time
from pykalman import KalmanFilter

# PyAutoGUI 설정
pyautogui.FAILSAFE = False

class VirtualMouse:
    def __init__(self, roi_x_min=0.3, roi_x_max=0.7, roi_y_min=0.3, roi_y_max=0.7):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_mx = self.screen_w // 2
        self.prev_my = self.screen_h // 2
        
        # Kalman Filter 초기화
        self.kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.eye(2),
            initial_state_mean=np.array([self.prev_mx, self.prev_my]),
            initial_state_covariance=np.eye(2),
            observation_covariance=np.eye(2) * 50.0,  # 노이즈 신뢰도 (값이 클수록 부드러워지나, 지연 발생)
            transition_covariance=np.eye(2) * 1.0     # 상태 변화 신뢰도
        )
        self.state_mean = np.array([self.prev_mx, self.prev_my])
        self.state_cov = np.eye(2)
        
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.roi_y_min = roi_y_min
        self.roi_y_max = roi_y_max
        
        self.last_click_time = 0

    def move(self, hx, hy):
        tx = ( (1 - hx) - self.roi_x_min ) / (self.roi_x_max - self.roi_x_min)
        ty = ( hy - self.roi_y_min ) / (self.roi_y_max - self.roi_y_min)
        tx, ty = np.clip(tx, 0, 1), np.clip(ty, 0, 1)
        
        raw_x = tx * self.screen_w
        raw_y = ty * self.screen_h
        
        # Kalman filter_update 적용 
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, np.array([raw_x, raw_y])
        )
        
        mx = int(self.state_mean[0])
        my = int(self.state_mean[1])
        
        pyautogui.moveTo(mx, my, _pause=False)
        self.prev_mx, self.prev_my = mx, my

    def click(self):
        current_time = time.time()
        # 0.5초 디바운스
        if current_time - self.last_click_time > 0.5:
            pyautogui.click()
            self.last_click_time = current_time
