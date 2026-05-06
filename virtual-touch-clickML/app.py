import os
import cv2
import time
import numpy as np
import json
from pathlib import Path
from collections import deque
from core.camera import get_camera
from core.mouse import VirtualMouse
from core.ui import UIManager
from core.kiosk import KioskUI
from detector.landmark import Detector

class InteractionApp:
    def __init__(self):
        # --- [수정] 카메라 인덱스 자동 검색 지원 ---
        # 환경 변수가 있으면 사용하고, 없으면 None을 넘겨 get_camera가 자동으로 찾게 함
        camera_index = os.environ.get('CAMERA_INDEX')
        if camera_index:
            camera_index = int(camera_index)
        else:
            camera_index = None # 자동 검색 모드

        self.camera = get_camera(width=1920, height=1080, fps=30, camera_index=camera_index)

        
        self.mouse = VirtualMouse(smoothing=0.5)
        self.detector = Detector()
        self.ui = UIManager()
        self.kiosk = KioskUI()
        
        self.current_mode = 'virtual touch' 
        self.all_strokes = []  
        self.active_stroke = []
        self.is_drawing = False
        
        self.view_rot_x = 0  
        self.view_rot_y = 0  
        self.view_zoom = 1.0
        self.prev_rot_pos = None
        self.prev_zoom_y = None

        # --- [드로잉 보정 관련] ---
        self.draw_smoothing = 0.4 
        self.sx, self.sy, self.sz = 0, 0, 0

        # --- [제스처 초기화 관련] ---
        self.clear_timer_start = None
        self.time_to_clear = 1.0 # 1초 유지 시 초기화

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
            if frame is None: continue
            
            display_image = cv2.flip(frame, 1)
            h_img, w_img = display_image.shape[:2]

            self.detector.process_hands(display_image)
            self.ui.draw_landmarks(display_image, self.detector, w_img, h_img)

            user_right = self.detector.get_hand_info('Right')
            user_left = self.detector.get_hand_info('Left')

            # --- [양손 주먹 제스처 초기화] ---
            if self.detector.is_hand_fist('Left') and self.detector.is_hand_fist('Right'):
                if self.clear_timer_start is None:
                    self.clear_timer_start = time.time()
                
                elapsed = time.time() - self.clear_timer_start
                # 시각적 피드백: 화면 중앙에 로딩 표시
                cv2.putText(display_image, f"CLEARING... {int((1-elapsed/self.time_to_clear)*100)}%", 
                            (w_img//2 - 150, h_img//2), 0, 1.5, (0, 0, 255), 3)
                
                if elapsed >= self.time_to_clear:
                    self.all_strokes = []
                    self.active_stroke = []
                    self.clear_timer_start = None
            else:
                self.clear_timer_start = None

            # 1. 사용자 왼손 (회전 및 줌)
            if user_left:
                lx, ly = user_left['pos']
                l_mid, l_thumb = user_left['middle'], user_left['thumb']
                l_dist_zoom = ((l_mid[0] - l_thumb[0])**2 + (l_mid[1] - l_thumb[1])**2 + (l_mid[2] - l_thumb[2])**2)**0.5
                
                # 왼손만 주먹일 때는 회전
                if self.detector.is_hand_fist('Left') and not self.detector.is_hand_fist('Right'):
                    if self.prev_rot_pos is None: self.prev_rot_pos = (lx, ly)
                    dx, dy = lx - self.prev_rot_pos[0], ly - self.prev_rot_pos[1]
                    self.view_rot_y += dx * 5.0
                    self.view_rot_x -= dy * 5.0
                    self.prev_rot_pos = (lx, ly)
                    cv2.putText(display_image, "ROTATION MODE", (50, 150), 0, 1, (255, 255, 0), 2)
                elif l_dist_zoom < 0.05:
                    if self.prev_zoom_y is None: self.prev_zoom_y = ly
                    dy = ly - self.prev_zoom_y
                    self.view_zoom = np.clip(self.view_zoom - dy * 2.0, 0.1, 5.0)
                    self.prev_zoom_y = ly
                    cv2.putText(display_image, f"ZOOM MODE: {self.view_zoom:.2f}x", (50, 150), 0, 1, (0, 255, 255), 2)
                    cv2.line(display_image, (int(l_mid[0]*w_img), int(l_mid[1]*h_img)), (int(l_thumb[0]*w_img), int(l_thumb[1]*h_img)), (0, 255, 255), 3)
                else:
                    self.prev_rot_pos, self.prev_zoom_y = None, None

            # 2. 사용자 오른손 (그리기)
            if user_right:
                hx, hy = user_right['pos']
                self.mouse.move(hx, hy)
                idx_pos, thumb_pos = user_right['index'], user_right['thumb']
                dist_3d = ((idx_pos[0] - thumb_pos[0])**2 + (idx_pos[1] - thumb_pos[1])**2 + (idx_pos[2] - thumb_pos[2])**2)**0.5
                lm = user_right['landmarks']
                hand_scale = ((lm[0].x - lm[5].x)**2 + (lm[0].y - lm[5].y)**2)**0.5
                z_val = np.clip((hand_scale - 0.05) / 0.2, 0, 1)
                raw_x, raw_y, raw_z = (idx_pos[0] + thumb_pos[0]) / 2, (idx_pos[1] + thumb_pos[1]) / 2, z_val

                if dist_3d < 0.08:
                    if not self.is_drawing:
                        self.sx, self.sy, self.sz = raw_x, raw_y, raw_z
                        self.is_drawing = True
                    else:
                        self.sx += (raw_x - self.sx) * self.draw_smoothing
                        self.sy += (raw_y - self.sy) * self.draw_smoothing
                        self.sz += (raw_z - self.sz) * self.draw_smoothing
                    
                    cx, cy, cz = 0.5, 0.5, 0.5
                    ix, iy, iz = (self.sx - cx)/self.view_zoom, (self.sy - cy)/self.view_zoom, (self.sz - cz)/self.view_zoom
                    ax, ay = -self.view_rot_x, -self.view_rot_y
                    iy_n = iy * np.cos(ax) - iz * np.sin(ax); iz_n = iy * np.sin(ax) + iz * np.cos(ax); iy, iz = iy_n, iz_n
                    ix_n = ix * np.cos(ay) - iz * np.sin(ay); iz_n = ix * np.sin(ay) + iz * np.cos(ay); ix, iz = ix_n, iz_n
                    
                    self.active_stroke.append((ix + cx, iy + cy, iz + cz))
                    cv2.circle(display_image, (int(self.sx*w_img), int(self.sy*h_img)), 15, (0, 255, 0), -1)
                    cv2.putText(display_image, "DRAWING", (50, 100), 0, 1, (0, 255, 0), 2)
                    line_color = (0, 255, 0)
                else:
                    if self.is_drawing and len(self.active_stroke) > 0:
                        self.all_strokes.append(self.active_stroke)
                        self.active_stroke = []
                    self.is_drawing = False
                    line_color = (0, 255, 255) if dist_3d < 0.12 else (0, 0, 255)

                cv2.line(display_image, (int(idx_pos[0]*w_img), int(idx_pos[1]*h_img)), (int(thumb_pos[0]*w_img), int(thumb_pos[1]*h_img)), line_color, 3)

            self.draw_strokes_3d(display_image, w_img, h_img)
            self.render_minimaps(display_image, w_img, h_img)

            output = cv2.resize(display_image, (w_img // 2, h_img // 2))
            cv2.imshow('3D Virtual Touch Painter', output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'): break
            elif key == ord('c') or key == ord('C'): self.all_strokes = []; self.active_stroke = []

    def draw_strokes_3d(self, img, w, h):
        strokes_to_draw = self.all_strokes + ([self.active_stroke] if self.active_stroke else [])
        ay, ax = self.view_rot_y, self.view_rot_x
        zoom, cx, cy, cz = self.view_zoom, 0.5, 0.5, 0.5
        for stroke in strokes_to_draw:
            if len(stroke) < 2: continue
            for i in range(1, len(stroke)):
                rotated_pts = []
                for p in [stroke[i-1], stroke[i]]:
                    x, y, z = (p[0] - cx) * zoom, (p[1] - cy) * zoom, (p[2] - cz) * zoom
                    x_n = x * np.cos(ay) - z * np.sin(ay); z_n = x * np.sin(ay) + z * np.cos(ay); x, z = x_n, z_n
                    y_n = y * np.cos(ax) - z * np.sin(ax); z_n = y * np.sin(ax) + z * np.cos(ax); y, z = y_n, z_n
                    rotated_pts.append((x + cx, y + cy, z + cz))
                p1, p2 = rotated_pts[0], rotated_pts[1]
                z_clamped = np.clip(p2[2], 0, 1)
                color = (int(255 * z_clamped), 50, int(255 * (1 - z_clamped)))
                thickness = int(max(1, 2 + p2[2] * 12))
                cv2.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, thickness)

    def render_minimaps(self, img, w, h):
        m_size = 250
        top_view = np.zeros((m_size, m_size, 3), dtype=np.uint8); side_view = np.zeros((m_size, m_size, 3), dtype=np.uint8)
        cv2.putText(top_view, "TOP (X-Z)", (10, 25), 0, 0.6, (255,255,255), 1); cv2.putText(side_view, "SIDE (Z-Y)", (10, 25), 0, 0.6, (255,255,255), 1)
        strokes_to_draw = self.all_strokes + ([self.active_stroke] if self.active_stroke else [])
        for stroke in strokes_to_draw:
            if len(stroke) < 2: continue
            for i in range(1, len(stroke)):
                p1, p2 = stroke[i-1], stroke[i]
                color = (int(255 * p2[2]), 100, int(255 * (1-p2[2])))
                cv2.line(top_view, (int(p1[0]*m_size), int((1-p1[2])*m_size)), (int(p2[0]*m_size), int((1-p2[2])*m_size)), color, 2)
                cv2.line(side_view, (int(p1[2]*m_size), int(p1[1]*m_size)), (int(p2[2]*m_size), int(p2[1]*m_size)), color, 2)
        try:
            img[20:20+m_size, w-m_size-20 : w-20] = top_view
            img[40+m_size:40+2*m_size, w-m_size-20 : w-20] = side_view
        except: pass
