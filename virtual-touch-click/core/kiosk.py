import cv2
import numpy as np

class KioskUI:
    def __init__(self, width=600, height=1200):
        self.width = width
        self.height = height
        self.window_name = "Kiosk UI"
        self.exit_requested = False
        
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
                
                # 버튼 텍스트 중앙 정렬
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
                tx = bx + (bw - text_size[0]) // 2
                ty = by + (bh + text_size[1]) // 2
                cv2.putText(img, name, (tx, ty), font, font_scale, (0, 0, 0), thickness)
            
        cv2.imshow(self.window_name, img)
