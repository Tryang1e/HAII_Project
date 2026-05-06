import cv2
import numpy as np

class KioskUI:
    def __init__(self, width=600, height=1200):
        self.width = width
        self.height = height
        self.window_name = "Kiosk UI"
        self.exit_requested = False
        self.hovered_button = None
        self.last_internal_click_time = 0
        
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
        self.button_pos = {
            "Exit": (10, 10, 280, 100),
            "Home": (310, 10, 280, 100),
            "1": (10, 130, 280, 180),
            "2": (310, 130, 280, 180),
            "3": (10, 330, 280, 180),
            "4": (310, 330, 280, 180),
            "Payment": (10, 530, 580, 140)
        }
        
        self.btn_height = self.height // len(self.buttons)
        
        # 윈도우 생성 및 콜백 등록
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (bx, by, bw, bh) in self.button_pos.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    print(f"[Kiosk] '{name}' button clicked!")
                    if name == "Exit":
                        self.exit_requested = True
                    break

    def set_hover(self, btn_name):
        self.hovered_button = btn_name

    def trigger_click(self, btn_name=None):
        import time
        current_time = time.time()
        # 0.5초 디바운스
        if current_time - self.last_internal_click_time < 0.5:
            return False
            
        target = btn_name if btn_name else self.hovered_button
        if target and target in self.button_pos:
            print(f"[Kiosk] '{target}' button clicked via Gesture & Pinch!")
            if target == "Exit":
                self.exit_requested = True
            self.last_internal_click_time = current_time
            return True
        return False

    def show(self):
        # Kiosk 배경 이미지 (검은색)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img = img + 50
        
        for name, color in self.buttons:
            if name in self.button_pos:
                bx, by, bw, bh = self.button_pos[name]
                
                is_hover = (self.hovered_button == name)
                
                # Highlight hovered button
                if is_hover:
                    bg_color = (min(color[0]+60, 255), min(color[1]+60, 255), min(color[2]+60, 255))
                    border_thickness = 4
                    edge_color = (50, 50, 255) # Reddish edge
                else:
                    bg_color = color
                    border_thickness = 2
                    edge_color = (50, 50, 50)
                
                # 버튼 배경
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), bg_color, -1)
                # 버튼 테두리
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), edge_color, border_thickness)
                
                # 가상의 마우스 커서/포인터 시각화
                if is_hover:
                    cx, cy = bx + bw // 2, by + bh // 2
                    cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
                    cv2.putText(img, "HOVER & PINCH", (bx + 10, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                
                # 버튼 텍스트 중앙 정렬
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
                tx = bx + (bw - text_size[0]) // 2
                ty = by + (bh + text_size[1]) // 2
                cv2.putText(img, name, (tx, ty), font, font_scale, (0, 0, 0), thickness)
            
        cv2.imshow(self.window_name, img)
