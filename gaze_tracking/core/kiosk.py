import cv2
import numpy as np
import time

class KioskUI:
    def __init__(self, screen_w=1920, screen_h=1080):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.width = screen_w
        self.height = screen_h
        self.window_name = "Kiosk Fullscreen"
        self.exit_requested = False
        
        # Pointer visual tracking
        self.cursor_x = -1
        self.cursor_y = -1
        self.dwell_ratio = 0.0
        
        # 버튼 정의: (텍스트, 색상(BGR))
        # 1:3 비율 (300x900)에 맞게 7개의 버튼 배치
        self.buttons = [
            ("Exit", (100, 100, 255)),
            ("Home", (255, 200, 200)),
            ("1", (230, 230, 230)),
            ("2", (230, 230, 230)),
            ("3", (230, 230, 230)),
            ("4", (230, 230, 230)),
            ("Payment", (200, 255, 200))
        ]

        # 버튼 이름으로 위치값(x, y, w, h)을 불러오는 딕셔너리
        
        # 버튼별 클릭 횟수 보관용 딕셔너리
        self.button_counts = {name: 0 for name, color in self.buttons}
        
        # 화면의 가로 길이에 맞춘 상대적 비율
        w_half = (self.width // 2) - 80
        btn_w_half = w_half
        btn_w_full = self.width - 100
        
        self.button_pos = {
            "Exit": (50, 50, btn_w_half, 150),
            "Home": (50 + w_half + 60, 50, btn_w_half, 150),
            "1": (50, 250, btn_w_half, 250),
            "2": (50 + w_half + 60, 250, btn_w_half, 250),
            "3": (50, 550, btn_w_half, 200),
            "4": (50 + w_half + 60, 550, btn_w_half, 200),
            "Payment": (50, 800, btn_w_full, 200)
        }
        
        self.btn_height = self.height // len(self.buttons)
        
        # 윈도우 생성 및 전체화면(풀스크린) 콜백 등록
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.trigger_click(x, y)

    def trigger_click(self, x, y):
        for name, (bx, by, bw, bh) in self.button_pos.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.button_counts[name] += 1
                print(f"[{time.strftime('%H:%M:%S')}] Kiosk '{name}' button clicked! (Total: {self.button_counts[name]})")
                if name == "Exit":
                    self.exit_requested = True
                break

    def update_indicator(self, mx, my, ratio):
        self.cursor_x = mx
        self.cursor_y = my
        self.dwell_ratio = ratio

    def show(self):
        # Kiosk 배경 이미지 (검은색)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img = img + 50
        
        for name, color in self.buttons:
            if name in self.button_pos:
                bx, by, bw, bh = self.button_pos[name]
                
                # 버튼 배경
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, -1)
                # 버튼 테두리
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (50, 50, 50), 2)
                
                # 버튼 텍스트와 카운트 수 중앙 정렬
                display_text = f"{name}" if self.button_counts[name] == 0 else f"{name} ({self.button_counts[name]})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
                tx = bx + (bw - text_size[0]) // 2
                ty = by + (bh + text_size[1]) // 2
                cv2.putText(img, display_text, (tx, ty), font, font_scale, (0, 0, 0), thickness)
            
        # Dwell Progress Indicator 그리기 (app.py에서 넘겨받은 값 그대로)
        if self.cursor_x >= 0 and self.cursor_y >= 0:
            # 시선 포인터 위치
            cv2.circle(img, (self.cursor_x, self.cursor_y), 5, (0, 255, 255), -1)
            
            # Dwell 게이지 원형 테두리
            if self.dwell_ratio > 0.05: # 약간의 시간이 지났을 때만 UI 표시
                angle = int(360 * self.dwell_ratio)
                cv2.ellipse(img, (self.cursor_x, self.cursor_y), (30, 30), 270, 0, angle, (0, 0, 255), 4)

        cv2.imshow(self.window_name, img)
