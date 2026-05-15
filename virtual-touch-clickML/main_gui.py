import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy, QGridLayout)
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QPoint, QEvent
from PySide6.QtGui import QImage, QPixmap, QFont, QMouseEvent, QShortcut, QKeySequence

from app import InteractionApp, HandState

class CameraThread(QThread):
    frame_ready = Signal(QImage, dict)
    
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.running = True
        
    def run(self):
        while self.running:
            frame = self.backend.process_frame(draw_ui=False)
            if frame is not None:
                state = {
                    'right_state': self.backend.right_state,
                    'is_rotation': getattr(self.backend, 'is_rotation', False),
                    'is_zoom': getattr(self.backend, 'is_zoom', False),
                    'is_ui_select': getattr(self.backend, 'is_ui_select', False),
                    'hand_detected': (self.backend.detector.hand_results and self.backend.detector.hand_results.multi_hand_landmarks),
                    'top_view_img': getattr(self.backend, 'top_view_img', None),
                    'side_view_img': getattr(self.backend, 'side_view_img', None),
                    'current_cursor_3d': self.backend.current_cursor_3d,
                    'cursor_x': self.backend.mouse.prev_mx,
                    'cursor_y': self.backend.mouse.prev_my,
                    'click': getattr(self.backend.mouse, 'click_event_triggered', False)
                }
                
                # Reset click flag so we don't click multiple times
                if self.backend.mouse.click_event_triggered:
                    self.backend.mouse.click_event_triggered = False
                
                # Convert to RGB here in the background thread to save UI time!
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Draw Crosshair
                fh, fw = rgb_frame.shape[:2]
                cx_line, cy_line = fw // 2, fh // 2
                cv2.line(rgb_frame, (cx_line - 20, cy_line), (cx_line + 20, cy_line), (100, 0, 0), 1)
                cv2.line(rgb_frame, (cx_line, cy_line - 20), (cx_line, cy_line + 20), (100, 0, 0), 1)
                
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.frame_ready.emit(qimg.copy(), state)
                
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Virtual Touch Painter")
        self.resize(1400, 800)
        
        # 전체 테마
        self.setStyleSheet("""
            QMainWindow { background-color: #0B0F19; }
            QLabel { color: #8F9BB3; font-family: 'Segoe UI', sans-serif; }
            QFrame#Panel { background-color: #131B2C; border-radius: 8px; }
            QFrame#TopBar { background-color: #131B2C; border-bottom: 1px solid #1C2740; }
            QFrame#LeftBar { background-color: #131B2C; border-right: 1px solid #1C2740; }
            QFrame#RightBar { background-color: #131B2C; border-left: 1px solid #1C2740; }
            QPushButton {
                background-color: transparent;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover { background-color: #1C2740; }
            QPushButton[active="true"] { background-color: #2F72FF; }
            QPushButton#BtnClear { background-color: #E21836; border-radius: 6px; }
            QPushButton#BtnClear:hover { background-color: #FF2A4D; }
            QPushButton.ColorBtn { border-radius: 6px; border: 2px solid transparent; }
            QPushButton.ColorBtn:hover { border: 2px solid #FFFFFF; }
            QPushButton.ColorBtn[active="true"] { border: 2px solid #FFFFFF; }
            
            QPushButton.SizeBtn { background-color: #1A2235; border-radius: 6px; font-size: 10px; color: white; }
            QPushButton.SizeBtn[active="true"] { background-color: #2F72FF; }
            QPushButton.SizeBtn:hover { background-color: #25304A; }
            
            QPushButton.IconButton { background-color: #1A2235; border-radius: 8px; font-size: 18px; }
            QPushButton.IconButton:hover { background-color: #25304A; }
            QPushButton.IconButton[active="true"] { background-color: #2F72FF; }
            
            QFrame.StatusBox { background-color: #25304A; border-radius: 10px; }
            QFrame.StatusBoxDark { background-color: #1A2235; border-radius: 10px; }
            
            QLabel.MinimapLabel { background-color: #1A2235; border-radius: 8px; }
        """)

        # 백엔드 엔진
        self.backend = InteractionApp()
        self.backend.camera.start()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 상단 툴바 (TopBar)
        self.setup_top_bar(main_layout)

        # 2. 중간 영역
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(0)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(mid_layout, 1)

        # 2-1. 좌측 사이드바 (LeftBar)
        self.setup_left_bar(mid_layout)

        # 2-2. 중앙 카메라 영역 (Center View)
        self.setup_center_view(mid_layout)

        # 2-3. 우측 미니맵 사이드바 (RightBar)
        self.setup_right_bar(mid_layout)

        # 가상 마우스 포인터 오버레이 (Floating Cursor)
        self.virtual_cursor = QLabel(self)
        self.virtual_cursor.setFixedSize(16, 16)
        self.virtual_cursor.setStyleSheet("""
            background-color: #2F72FF; 
            border: 2px solid white; 
            border-radius: 8px;
        """)
        self.virtual_cursor.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.virtual_cursor.hide() # 손이 없으면 숨김

        # 타이머 (윈도우 위치 업데이트 용도)
        self.timer = QTimer()
        self.timer.timeout.connect(self.sync_window_rect)
        self.timer.start(100)

        # 키보드 단축키 (QShortcut) 설정 - IME 상태와 무관하게 동작
        QShortcut(QKeySequence("U"), self).activated.connect(self.do_undo)
        QShortcut(QKeySequence("R"), self).activated.connect(self.do_redo)
        QShortcut(QKeySequence("C"), self).activated.connect(self.do_clear)
        QShortcut(QKeySequence("P"), self).activated.connect(self.toggle_mode)

        # 카메라 쓰레드
        self.camera_thread = CameraThread(self.backend)
        self.camera_thread.frame_ready.connect(self.update_ui)
        self.camera_thread.start()

    def sync_window_rect(self):
        pos = self.mapToGlobal(self.rect().topLeft())
        self.backend.mouse.set_window_rect(pos.x(), pos.y(), self.width(), self.height())

    def simulate_click(self, global_x, global_y):
        # 마우스 포인터 위젯 자체를 클릭하는 것을 방지하기 위해 잠깐 숨김
        self.virtual_cursor.hide()
        widget = QApplication.widgetAt(global_x, global_y)
        self.virtual_cursor.show()
        
        if widget is not None:
            # Add visual feedback on cursor
            self.virtual_cursor.setStyleSheet("background-color: #E21836; border: 2px solid white; border-radius: 8px;")
            QTimer.singleShot(150, lambda: self.virtual_cursor.setStyleSheet("background-color: #2F72FF; border: 2px solid white; border-radius: 8px;"))
            
            local_pos = widget.mapFromGlobal(QPoint(global_x, global_y))
            press = QMouseEvent(QEvent.MouseButtonPress, local_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            release = QMouseEvent(QEvent.MouseButtonRelease, local_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            QApplication.postEvent(widget, press)
            QApplication.postEvent(widget, release)

    def setup_top_bar(self, parent_layout):
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_bar.setFixedHeight(60)
        t_layout = QHBoxLayout(top_bar)
        t_layout.setContentsMargins(15, 10, 15, 10)
        t_layout.setSpacing(10)

        self.btn_brush = QPushButton("🖌️")
        self.btn_brush.setProperty("active", True)
        self.btn_brush.setFixedSize(40, 40)
        self.btn_brush.setStyleSheet("font-size: 20px;")
        
        self.btn_eraser = QPushButton("🧽")
        self.btn_eraser.setFixedSize(40, 40)
        self.btn_eraser.setStyleSheet("font-size: 20px;")
        
        t_layout.addWidget(self.btn_brush)
        t_layout.addWidget(self.btn_eraser)
        
        sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine); sep1.setStyleSheet("color: #1C2740;")
        t_layout.addWidget(sep1)
        
        t_layout.addWidget(QLabel("색상:"))
        
        colors = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080"]
        for i, c in enumerate(colors):
            btn = QPushButton()
            btn.setFixedSize(30, 30)
            btn.setStyleSheet(f"background-color: {c};")
            btn.setProperty("class", "ColorBtn")
            if i == 2: btn.setProperty("active", True)
            t_layout.addWidget(btn)
            
        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine); sep2.setStyleSheet("color: #1C2740;")
        t_layout.addWidget(sep2)
        
        t_layout.addWidget(QLabel("굵기:"))
        
        sizes = ["·", "•", "●", "⬤", "⬤", "⬤", "⬤"]
        for i, s in enumerate(sizes):
            btn = QPushButton(s)
            btn.setFixedSize(30, 30)
            btn.setProperty("class", "SizeBtn")
            if i == 1: btn.setProperty("active", True)
            t_layout.addWidget(btn)

        t_layout.addStretch()

        self.btn_undo = QPushButton("↩")
        self.btn_undo.setProperty("class", "IconButton")
        self.btn_undo.setFixedSize(40, 40)
        
        self.btn_redo = QPushButton("↪")
        self.btn_redo.setProperty("class", "IconButton")
        self.btn_redo.setFixedSize(40, 40)
        
        self.btn_clear = QPushButton("↻")
        self.btn_clear.setObjectName("BtnClear")
        self.btn_clear.setFixedSize(40, 40)
        self.btn_clear.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        self.btn_undo.clicked.connect(self.do_undo)
        self.btn_redo.clicked.connect(self.do_redo)
        self.btn_clear.clicked.connect(self.do_clear)

        t_layout.addWidget(self.btn_undo)
        t_layout.addWidget(self.btn_redo)
        t_layout.addWidget(self.btn_clear)

        parent_layout.addWidget(top_bar)

    def setup_left_bar(self, parent_layout):
        left_bar = QFrame()
        left_bar.setObjectName("LeftBar")
        left_bar.setFixedWidth(60)
        l_layout = QVBoxLayout(left_bar)
        l_layout.setContentsMargins(10, 20, 10, 20)
        l_layout.setSpacing(15)

        icons = ["▦", "⤡", "✋", "⛶"]
        for i, ic in enumerate(icons):
            btn = QPushButton(ic)
            btn.setProperty("class", "IconButton")
            btn.setFixedSize(40, 40)
            if i == 1: 
                self.btn_mode = btn
                btn.setProperty("active", self.backend.is_2d_mode)
                btn.clicked.connect(self.toggle_mode)
            l_layout.addWidget(btn)
            
        l_layout.addStretch()
        parent_layout.addWidget(left_bar)

    def setup_center_view(self, parent_layout):
        self.center_container = QWidget()
        self.center_container.setStyleSheet("background-color: #0B0F19;")
        c_layout = QGridLayout(self.center_container)
        c_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        c_layout.addWidget(self.video_label, 0, 0)
        
        overlay_widget = QWidget()
        overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        o_layout = QVBoxLayout(overlay_widget)
        o_layout.setContentsMargins(20, 20, 0, 0)
        o_layout.setSpacing(10)
        
        self.lbl_action_status = QLabel("  ▶ 대기 중")
        self.lbl_action_status.setProperty("class", "StatusBox")
        self.lbl_action_status.setFixedSize(120, 40)
        self.lbl_action_status.setStyleSheet("color: white; font-weight: bold; font-size: 14px; padding-left: 10px;")
        
        self.lbl_hand_status = QLabel("  ✋ 손 감지 안됨")
        self.lbl_hand_status.setProperty("class", "StatusBoxDark")
        self.lbl_hand_status.setFixedSize(140, 40)
        self.lbl_hand_status.setStyleSheet("padding-left: 10px;")
        
        o_layout.addWidget(self.lbl_action_status)
        o_layout.addWidget(self.lbl_hand_status)
        o_layout.addStretch()
        
        c_layout.addWidget(overlay_widget, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        parent_layout.addWidget(self.center_container, 1)

    def setup_right_bar(self, parent_layout):
        right_bar = QFrame()
        right_bar.setObjectName("RightBar")
        right_bar.setFixedWidth(280)
        r_layout = QVBoxLayout(right_bar)
        r_layout.setContentsMargins(20, 20, 20, 20)
        r_layout.setSpacing(10)
        
        lbl_top = QLabel("탑 뷰 (Top View)")
        lbl_top.setStyleSheet("color: #B0BEC5;")
        self.top_minimap = QLabel()
        self.top_minimap.setProperty("class", "MinimapLabel")
        self.top_minimap.setFixedSize(240, 160)
        self.top_minimap.setAlignment(Qt.AlignCenter)
        lbl_x = QLabel("X ? 평면")
        lbl_x.setStyleSheet("color: #607D8B; font-size: 11px;")
        
        r_layout.addWidget(lbl_top)
        r_layout.addWidget(self.top_minimap)
        r_layout.addWidget(lbl_x)
        r_layout.addSpacing(20)
        
        lbl_side = QLabel("사이드 뷰 (Side View)")
        lbl_side.setStyleSheet("color: #B0BEC5;")
        self.side_minimap = QLabel()
        self.side_minimap.setProperty("class", "MinimapLabel")
        self.side_minimap.setFixedSize(240, 160)
        self.side_minimap.setAlignment(Qt.AlignCenter)
        lbl_z = QLabel("Z-Y 평면")
        lbl_z.setStyleSheet("color: #607D8B; font-size: 11px;")
        
        r_layout.addWidget(lbl_side)
        r_layout.addWidget(self.side_minimap)
        r_layout.addWidget(lbl_z)
        r_layout.addSpacing(20)
        
        coord_frame = QFrame()
        coord_frame.setProperty("class", "Panel")
        coord_frame.setFixedHeight(90)
        c_lay = QVBoxLayout(coord_frame)
        c_lay.setContentsMargins(15, 15, 15, 15)
        
        self.lbl_x_val = QLabel("X:                             0.00")
        self.lbl_y_val = QLabel("Y:                             0.00")
        self.lbl_z_val = QLabel("Z:                             0.00")
        for lbl in [self.lbl_x_val, self.lbl_y_val, self.lbl_z_val]:
            lbl.setStyleSheet("color: white; font-family: 'Consolas', monospace;")
            c_lay.addWidget(lbl)
            
        r_layout.addWidget(coord_frame)
        r_layout.addStretch()

        parent_layout.addWidget(right_bar)

    def do_undo(self):
        if self.backend.all_strokes:
            self.backend.undone_strokes.append(self.backend.all_strokes.pop())
            
    def do_redo(self):
        if self.backend.undone_strokes:
            self.backend.all_strokes.append(self.backend.undone_strokes.pop())
            
    def do_clear(self):
        while self.backend.all_strokes:
            self.backend.undone_strokes.append(self.backend.all_strokes.pop())
        self.backend.active_stroke.clear()
        
    def toggle_mode(self):
        self.backend.is_2d_mode = not self.backend.is_2d_mode
        if hasattr(self, 'btn_mode') and self.btn_mode:
            self.btn_mode.setProperty("active", self.backend.is_2d_mode)
            self.btn_mode.style().unpolish(self.btn_mode)
            self.btn_mode.style().polish(self.btn_mode)

    def update_ui(self, qimg, state):
        # 1. 카메라 프레임 업데이트 (FastTransformation으로 최적화)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        
        # 2. 가상 마우스 포인터 렌더링 및 클릭
        if state['hand_detected']:
            self.virtual_cursor.show()
            global_x = state['cursor_x']
            global_y = state['cursor_y']
            
            # 윈도우 로컬 좌표로 변환
            win_pos = self.mapFromGlobal(QPoint(global_x, global_y))
            
            # 화면 크기를 벗어나면 윈도우 안으로 클리핑 (이미 Backend에서 클리핑하지만 안전하게 한 번 더)
            local_x = max(0, min(self.width() - 20, win_pos.x()))
            local_y = max(0, min(self.height() - 20, win_pos.y()))
            
            # Move cursor
            self.virtual_cursor.move(local_x, local_y)
            self.virtual_cursor.raise_()
            
            if state['click']:
                self.simulate_click(global_x, global_y)
        else:
            self.virtual_cursor.hide()

        # 3. 상태 오버레이 업데이트
        if state['right_state'] == HandState.DRAWING:
            self.lbl_action_status.setText("  ✏️ 그리는 중")
            self.lbl_action_status.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold; font-size: 14px; border-radius: 10px; padding-left: 10px;")
        elif state['is_rotation']:
            self.lbl_action_status.setText("  🔄 회전 중")
            self.lbl_action_status.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; font-size: 14px; border-radius: 10px; padding-left: 10px;")
        elif state['is_zoom']:
            self.lbl_action_status.setText("  🔍 줌 인/아웃")
            self.lbl_action_status.setStyleSheet("background-color: #eab308; color: black; font-weight: bold; font-size: 14px; border-radius: 10px; padding-left: 10px;")
        elif state['is_ui_select']:
            self.lbl_action_status.setText("  🖱️ UI 선택 중")
            self.lbl_action_status.setStyleSheet("background-color: #d946ef; color: white; font-weight: bold; font-size: 14px; border-radius: 10px; padding-left: 10px;")
        else:
            self.lbl_action_status.setText("  ▶ 대기 중")
            self.lbl_action_status.setStyleSheet("background-color: #25304A; color: white; font-weight: bold; font-size: 14px; border-radius: 10px; padding-left: 10px;")
            
        if state['hand_detected']:
            self.lbl_hand_status.setText("  ✋ 손 감지됨")
            self.lbl_hand_status.setStyleSheet("background-color: #314060; color: white; border-radius: 10px; padding-left: 10px;")
        else:
            self.lbl_hand_status.setText("  ✋ 손 감지 안됨")
            self.lbl_hand_status.setStyleSheet("background-color: #1A2235; color: #8F9BB3; border-radius: 10px; padding-left: 10px;")
        
        # 4. 미니맵 업데이트
        if state['top_view_img'] is not None:
            tm = cv2.cvtColor(state['top_view_img'], cv2.COLOR_BGR2RGB)
            th, tw, tc = tm.shape
            q_tm = QImage(tm.data, tw, th, tc*tw, QImage.Format_RGB888)
            self.top_minimap.setPixmap(QPixmap.fromImage(q_tm).scaled(self.top_minimap.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
            
        if state['side_view_img'] is not None:
            sm = cv2.cvtColor(state['side_view_img'], cv2.COLOR_BGR2RGB)
            sh, sw, sc = sm.shape
            q_sm = QImage(sm.data, sw, sh, sc*sw, QImage.Format_RGB888)
            self.side_minimap.setPixmap(QPixmap.fromImage(q_sm).scaled(self.side_minimap.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
            
        # 5. 좌표 업데이트
        if state['current_cursor_3d']:
            cx, cy, cz = state['current_cursor_3d']
            self.lbl_x_val.setText(f"X: {cx:.2f}".ljust(35))
            self.lbl_y_val.setText(f"Y: {cy:.2f}".ljust(35))
            self.lbl_z_val.setText(f"Z: {cz:.2f}".ljust(35))

    def closeEvent(self, event):
        self.timer.stop()
        self.camera_thread.stop()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
