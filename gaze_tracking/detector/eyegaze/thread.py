import os
import cv2
import threading
import onnxruntime
onnxruntime.set_default_logger_severity(3)  # ERROR만 표시 (Warning 숨김)
import numpy as np
from collections import deque
import time

from detector.timer.timer import Timer
from detector.eyegaze import face_utils
from detector.eyegaze.gaze_tracker import gaze_tracker
          

class EyeGazeThread(threading.Thread):
    def __init__(self, server, display=False, use_thread=True, display_size=[1920,1080], \
        top_margin=0, bottom_margin=0, left_margin=0, right_margin=0, auto_cali=False, sr_selection="left", gaze_mode="head"):
        threading.Thread.__init__(self)

        self.gaze_mode = gaze_mode
        self.setDaemon(True)
        self.use_thread = use_thread
        self.frame_name = "EyeGaze"
        self.is_close = False
        self.display = display
        self.server = server
        self.frame = None
        self.auto_cali = auto_cali
        self.frame_lock = threading.Lock()
        self.display_size = display_size
        self.eye_xy_pts = None
        self.face_frame = [640, 512]
        self.margined_top = top_margin
        self.margined_bottom = display_size[1] - bottom_margin
        self.margined_left = left_margin
        self.margined_right = display_size[0] - right_margin
        
        self.margined_width = self.margined_right - self.margined_left
        self.margined_height = self.margined_bottom - self.margined_top
        
        self.margined_cx = int(self.margined_left + self.margined_width/2.0)
        self.margined_cy = int(self.margined_top + self.margined_height/2.0)
        
        base_dir = os.path.join(os.path.dirname(__file__), '../../')
        if os.path.exists(os.path.join(base_dir, 'model/TDDFA.onnx')):
            self.lm_session = onnxruntime.InferenceSession(os.path.join(base_dir, 'model/TDDFA.onnx'), None)
            self.face_session = onnxruntime.InferenceSession(os.path.join(base_dir, 'model/face_detector.onnx'), None)
        else:
            self.lm_session = onnxruntime.InferenceSession('../model/TDDFA.onnx', None)
            self.face_session = onnxruntime.InferenceSession('../model/face_detector.onnx', None)
            
        self.gaze_tracker = gaze_tracker(self.display_size, sr_selection, server)
        self.gaze_x = -1
        self.gaze_y = -1
        self.fps = 0
        self.seat_position = 'none'
        self.flm_vers = None
        self.flm_que = deque()
        self.xdeg = 0
        self.ydeg = 0
        self.zdeg = 0
        self.frame_cnt = 0
        
        # For draw
        self.size_factor = 22
        self.line_size = int(self.size_factor / 4)
        
    def process(self, frame, width, height, eye_selection):
        Timer.set_eyegaze_time_stamp()
        diff_cx, diff_cy = -1, -1
        self.frame_cnt += 1
    
        # face detection
        if len(frame.shape) == 2:
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.face_session.run(None, {"input": image})
        boxes, _, _ = face_utils.predict(frame.shape[1], frame.shape[0], confidences, boxes, 0.45)
        
        if len(boxes) > 0:
            box = boxes[0, :]
            box_margin = int((box[2] - box[0])/6)
            # add box margin for accurate face landmark inference
            sx = box[0]-box_margin if box[0]-box_margin > 0 else 0
            sy = box[1]-box_margin if box[1]-box_margin > 0 else 0
            ex = box[2]+box_margin if box[2]+box_margin < width else width-1
            ey = box[3]+box_margin if box[3]+box_margin < height else height-1
            
            # facial landmark detection
            if len(frame.shape) == 2:
                cropped_img = cv2.cvtColor(frame[sy:ey, sx:ex].copy(), cv2.COLOR_GRAY2RGB)
            else:
                cropped_img = cv2.cvtColor(frame[sy:ey, sx:ex].copy(), cv2.COLOR_BGR2RGB)
            
            flm_vers, head_vers_3d = face_utils.TDDFA_detector(self.lm_session, cropped_img, sx, sy, ex, ey)
            (self.xdeg, self.ydeg, self.zdeg), _ = face_utils.headgaze_detector(head_vers_3d)
            
            # try:
            #     if len(self.flm_vers) == len(flm_vers):
            #         diff = abs(np.sum(self.flm_vers - flm_vers))
            #         if diff < 300:
            #             self.flm_que.append(flm_vers.copy())
            #         if diff > 500:
            #             self.flm_que.append(flm_vers.copy())
            #     else:
            #         self.flm_que.append(flm_vers.copy())
            # except Exception as e:
            #     self.flm_que.append(flm_vers.copy())
            #     pass
            
            self.flm_que.append(flm_vers.copy())

            # smoothing: enqueue and dequeue ops
            if len(self.flm_que) >= int(self.fps/4+1):
                self.flm_vers = np.mean(self.flm_que, axis=0).astype(np.float32)
                self.flm_que.popleft()

                # cropped right eye ROI
                if eye_selection == "right":
                    eye_img, eye_lm_info = face_utils.get_right_eye_roi(frame, self.flm_vers)
                else:
                    eye_img, eye_lm_info = face_utils.get_left_eye_roi(frame, self.flm_vers)
                    
                if len(eye_img.shape) == 3:
                    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                
                # sr, pupil detection function
                # [gaze_tracker.tracking 파라미터 정보 및 설명]
                # 1. frame (np.ndarray): 원본 해상도의 현재 프레임 이미지
                # 2. eye_img (np.ndarray): 랜드마크 기반으로 크롭된 눈 영역(ROI) 흑백 이미지
                # 3. line_size (int): UI 시각화 십자선(Crosshair)의 길이
                # 4. fps (float/int): 현재 연산 기준 초당 프레임 수(FPS)
                # 5. margin_sy (int): 화면 시각화 출력을 위한 상단 여백 (y축 시작점)
                # 6. margin_ey (int): 화면 시각화 출력을 위한 하단 여백 (y축 종료점)
                # 7. margin_sx (int): 화면 시각화 출력을 위한 좌측 여백 (x축 시작점)
                # 8. margin_ex (int): 화면 시각화 출력을 위한 우측 여백 (x축 종료점)
                # 9. display (bool): 시각화 화면(UI) 출력 여부 (True / False)
                # 10. face_frame (list): 얼굴을 보여주기 위한 내부 UI 해상도 리스트 [가로, 세로]
                # 11. xdeg (float): 머리 방향의 X축 회전 각도 (Pitch - 상하 고개 기울기)
                # 12. ydeg (float): 머리 방향의 Y축 회전 각도 (Yaw - 좌우 고개 방향)
                # 13. eye_lm_info (list): 크롭된 눈 영역의 좌상단 오프셋 및 기준 랜드마크 좌표 [sx, sy, ref_x, ref_y]
                # 14. gaze_mode (str): 사용자의 시선 산출 모드 선택 ("eye", "head", "combine" 중 택1)
                # 15. head_loc (tuple): 얼굴 랜드마크 68개의 무게중심(Center of Mass) 좌표 (x, y)
                head_com_x = np.mean(self.flm_vers[:, 0])
                head_com_y = np.mean(self.flm_vers[:, 1])
                
                self.gaze_tracker.tracking(frame, eye_img, self.line_size, self.fps, self.margined_top,\
                    self.margined_bottom,self.margined_left, self.margined_right, \
                        self.display, self.face_frame, self.xdeg, self.ydeg, eye_lm_info, \
                            self.gaze_mode, (head_com_x, head_com_y))
                
                self.gaze_x, self.gaze_y, diff_cx, diff_cy = self.gaze_tracker.get_gazedata(self.fps, self.gaze_mode)

                # clipping by gaze inference's boundery
                if self.gaze_y < self.margined_top + int(0.05*(self.margined_height)):
                    self.gaze_y = self.margined_top + int(0.05*(self.margined_height))
                if self.gaze_y > self.margined_bottom - int(0.05*(self.margined_height)):
                    self.gaze_y = self.margined_bottom - int(0.05*(self.margined_height))
                if self.gaze_x < self.margined_left + int(0.05*self.margined_width):
                    self.gaze_x = self.margined_left + int(0.05*self.margined_width)
                if self.gaze_x > self.margined_left + int(0.95*self.margined_width):
                    self.gaze_x = self.margined_left + int(0.95*self.margined_width)
                    
                for i in range(0,68):
                    cv2.circle(frame, (int(round(flm_vers[i,0])),int(round(flm_vers[i,1]))), 2, 255, -1, cv2.LINE_AA)
        else:
            diff_cx, diff_cy = -1, -1 
            
            print('[W:6] Failed facial landmark tracking process (eyegaze/thread.py 131 line)')
        
        self.fps = Timer.get_eyegaze_fps()
            
        # Visualize
        if self.display:
            try:
                vis_image = np.ones((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)*30
                frame = cv2.flip(frame, 1)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                try:
                    if self.display_size[0] == 1920:
                        temp_frame = cv2.resize(frame, (self.face_frame[0], self.face_frame[1]), interpolation=cv2.INTER_CUBIC)
                        vis_image[self.margined_cy-int(self.face_frame[1]/2):self.margined_cy+int(self.face_frame[1]/2),
                            self.margined_cx-int(self.face_frame[0]/2):self.margined_cx+int(self.face_frame[0]/2)] = temp_frame
                    else:
                        temp_frame = cv2.resize(frame, (self.face_frame[0], self.face_frame[1]), interpolation=cv2.INTER_CUBIC)
                        vis_image[self.margined_cy-int(self.face_frame[1]/2):self.margined_cy+int(self.face_frame[1]/2),
                            self.margined_cx-int(self.face_frame[0]/2):self.margined_cx+int(self.face_frame[0]/2)] = temp_frame
                except Exception as e:
                    print('[W:3] Failed create display Image (eyegaze/thread.py 224 line)',str(e))
                    if self.display_size[0] == 1920:
                        temp_frame = cv2.resize(frame, (self.face_frame[0], self.face_frame[1]), interpolation=cv2.INTER_CUBIC)
                        vis_image[self.margined_cy-int(self.face_frame[1]/2):self.margined_cy+int(self.face_frame[1]/2),
                            self.margined_cx-int(self.face_frame[0]/2):self.margined_cx+int(self.face_frame[0]/2)] = temp_frame
                    else:
                        temp_frame = cv2.resize(frame, (self.face_frame[0], self.face_frame[1]), interpolation=cv2.INTER_CUBIC)
                        vis_image[self.margined_cy-int(self.face_frame[1]/2):self.margined_cy+int(self.face_frame[1]/2),
                            self.margined_cx-int(self.face_frame[0]/2):self.margined_cx+int(self.face_frame[0]/2)] = temp_frame
                
                vis_image[:self.margined_top,:] = (55,55,55)
                vis_image[self.margined_bottom:,:] = (55,55,55)
                vis_image[:,:self.margined_left] = (55,55,55)
                vis_image[:,self.margined_right:] = (55,55,55)
                cv2.putText(vis_image, "%02d fps" % round(self.fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(vis_image, "x_rot, y_rot: %d, %d" % (self.xdeg, self.ydeg), (10, 50), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(vis_image, "diffx, diffy: %0.4f, %0.4f" % (diff_cx, diff_cy), (10, 80), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if self.gaze_x is not None:
                    cv2.circle(vis_image, (self.gaze_x, self.gaze_y), int(self.size_factor* 2.2) , \
                        (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.line(vis_image, (self.gaze_x - self.line_size, self.gaze_y), \
                        (self.gaze_x + self.line_size, self.gaze_y), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(vis_image, (self.gaze_x, self.gaze_y - self.line_size), \
                        (self.gaze_x, self.gaze_y + self.line_size), (0, 255, 0), 2, cv2.LINE_AA)
                
                vis_image = cv2.resize(vis_image, (self.display_size[0], self.display_size[1]), interpolation=cv2.INTER_CUBIC)
                
                if self.display and self.gaze_tracker.mode != "calibration":
                    cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self.frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow(self.frame_name, vis_image)

                if self.use_thread:
                    key = cv2.waitKey(1)
                    if key == 27:
                        self.is_close = True
            except Exception as e:
                print('[W:4] Displya error (eyegaze/thread.py 253 line)',str(e))
