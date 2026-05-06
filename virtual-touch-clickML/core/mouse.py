import pyautogui
import numpy as np
import time

# PyAutoGUI 설정
pyautogui.FAILSAFE = False

class VirtualMouse:
    def __init__(self, smoothing=0.5, roi_x_min=0.3, roi_x_max=0.7, roi_y_min=0.3, roi_y_max=0.7):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_mx = self.screen_w // 2
        self.prev_my = self.screen_h // 2
        self.smoothing = smoothing
        
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.roi_y_min = roi_y_min
        self.roi_y_max = roi_y_max
        
        self.last_click_time = 0

    def move(self, hx, hy):
        tx = ( (1 - hx) - self.roi_x_min ) / (self.roi_x_max - self.roi_x_min)
        ty = ( hy - self.roi_y_min ) / (self.roi_y_max - self.roi_y_min)
        tx, ty = np.clip(tx, 0, 1), np.clip(ty, 0, 1)
        
        mx = int(self.prev_mx + (int(tx * self.screen_w) - self.prev_mx) * self.smoothing)
        my = int(self.prev_my + (int(ty * self.screen_h) - self.prev_my) * self.smoothing)
        pyautogui.moveTo(mx, my, _pause=False)
        self.prev_mx, self.prev_my = mx, my

    def click(self):
        current_time = time.time()
        # 0.5초 디바운스
        if current_time - self.last_click_time > 0.5:
            pyautogui.click()
            self.last_click_time = current_time
