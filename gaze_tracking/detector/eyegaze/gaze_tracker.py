import os
import cv2
import numpy as np
import time
from detector.eyegaze.PupilDetector import GradientIntersect

class gaze_tracker():
    """
    This is the main class used to run a model.
    """
    def __init__(self, display_size=[1920,1080], sr_selection="left", server=None):
            
        base_dir = os.path.join(os.path.dirname(__file__), '../../')
                
        if os.path.exists(os.path.join(base_dir, 'model/cali_panel.PNG')):
            self.cali_panel = cv2.imread(os.path.join(base_dir, 'model/cali_panel.PNG'))
        
        if os.path.exists('../model/cali_panel.PNG'):
            self.cali_panel = cv2.imread('../model/cali_panel.PNG')
            
        self.gaze_xs = []
        self.gaze_ys = []
        self.server = server
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.eye_width = 120
        self.eye_height = 80
        self.eye_cx = int(self.eye_width/2)
        self.eye_cy = int(self.eye_height/2)
        
        self.mode = "tracking"
        self.sr_failed = False
        self.gi = GradientIntersect()
            
        self.disp_width = display_size[0]
        self.disp_height = display_size[1]
        self.eye_xdeg = []
        self.eye_ydeg = []
        self.anchor_eye_x = 0
        self.anchor_eye_y = 0
        self.eye_x = 0
        self.eye_y = 0
        self.timer = 0
        self.time_step = 4.0
        self.begin = time.time() + self.time_step - 1.2
        self.cx = 0 # pupil_center x
        self.cy = 0 # pupil_center y
        
        self.pupil_init = True
        self.pupil_values = []
        self.pupil_value = 90
        self.ref_ratios = []
        self.ref_sizes = []
        self.avg_ratio = 1.0
        self.avg_size = 120
        
        self.eye0 = []
        self.eye_cali0 = []
        self.eye_cali1 = []
        self.eye_cali2 = []
        self.eye_cali3 = []
        self.eye_cali4 = []
        self.end = 0
        self.eye_img_list = []
        self.eye_PM = None
        
    def change_pupil(self, pos):
        self.pupil_th = int(pos)
    
    def calibration(self):
        self.anchor_eye_x = 0
        self.anchor_eye_y = 0
        self.eye0 = []
        self.gaze_xs = []
        self.gaze_ys = []
        self.eye_xdeg = []
        self.eye_ydeg = []
        self.eye_cali0 = []
        self.eye_cali1 = []
        self.eye_cali2 = []
        self.eye_cali3 = []
        self.eye_cali4 = []
        self.begin = time.time() + self.time_step - 1.2
        self.eye_PM = None
        self.end = 0
        self.cx = 0 # pupil_center x
        self.cy = 0 # pupil_center y
        self.pupil_init = True
        self.pupil_values = []
        self.pupil_value = 90
        self.ref_ratios = []
        self.ref_sizes = []
        self.avg_ratio = 1.0
        self.avg_size = 120
        self.eye_img_list = []

    def draw_cali_circle(self, window_img, center_pt, size_factor, line_size, timer, time_step):
        cv2.circle(window_img, center_pt, int(size_factor) +1, (255,0,0), 3, cv2.LINE_AA)
        cv2.circle(window_img, center_pt, int((size_factor/self.time_step)*(timer-time_step)), (0,0,255), -1, cv2.LINE_AA)
        cv2.line(window_img, (center_pt[0] - line_size, center_pt[1]), (center_pt[0] + line_size, center_pt[1]), (0,255,0), 2, cv2.LINE_AA)
        cv2.line(window_img, (center_pt[0], center_pt[1] - line_size), (center_pt[0], center_pt[1] + line_size), (0,255,0), 2, cv2.LINE_AA)
        
    def pupil_detection(self, img, fps):
        pupil_img = cv2.medianBlur(img, 5)
        self.eye_img_list.append(pupil_img.copy())
        
        if len(self.eye_img_list)>=3:
            while len(self.eye_img_list)>3:
                self.eye_img_list.pop(0)
                
        avg_image = self.eye_img_list[0].copy()
        for i in range(1, len(self.eye_img_list)):
            avg_image = cv2.addWeighted(self.eye_img_list[i], 0.4, avg_image, 0.6, 0.0)
        
        avg_image = cv2.blur(avg_image,(3,3))
        
        # Mask out the eyebrow region (top 30% of the image) to prevent false detection
        eyebrow_margin = int(self.eye_height * 0.3)
        avg_image[:eyebrow_margin, :] = 255
                
        cx, cy = -1, -1
        
        # 1. Intensity minimum thresholding
        # Pupil is the absolute darkest region. Fixed percentiles swallow the iris.
        # Instead, find the darkest pixel and threshold tightly around it.
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(avg_image)
        
        # Dynamic threshold: darkest value + 10% of dynamic range, or a fixed small offset
        # This prevents picking up the lighter iris if the pupil exists.
        threshold_val = min_val + max((max_val - min_val) * 0.01, 1)
        
        _, thresh = cv2.threshold(avg_image, threshold_val, 255, cv2.THRESH_BINARY_INV)
        
        # # 2. Morphological operations
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        
        # 3. Contour Detection & Circularity validation
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 1200:
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # The pupil shape is generally highly circular. 
            if 0.4 < circularity < 1.3:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    temp_cx = int(M["m10"] / M["m00"])
                    temp_cy = int(M["m01"] / M["m00"])
                    
                    # Score by circularity and darkness
                    darkness = 255 - avg_image[temp_cy, temp_cx]
                    score = circularity * 100 + darkness
                    
                    if score > best_score:
                        best_score = score
                        cx = temp_cx / self.eye_width
                        cy = temp_cy / self.eye_height

        if cx > 0 and cy > 0:
            self.cx = cx
            self.cy = cy
        else:
            # Fallback to the absolute darkest spot if strict circle isn't found
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(avg_image)
            cx = min_loc[0] / self.eye_width
            cy = min_loc[1] / self.eye_height
            self.cx = cx
            self.cy = cy

        return cx, cy, thresh

    
    #################################### pupil, sr detecion proceess ##################################
    def tracking(self, frame, eye_img, line_size, fps, margin_sy, margin_ey, margin_sx, margin_ex, \
        display, face_frame, x_rot, y_rot, eye_lm_info, gaze_mode, head_loc):

        if self.timer < -self.time_step:
            self.begin = time.time() + self.time_step - 1.2
            self.end = time.time()
        
        self.end = time.time()
        self.timer = self.end - self.begin
        eye_sx, eye_sy, ref_lmx, ref_lmy = eye_lm_info
        h, w = eye_img.shape
        # resize eye ROI
        img = cv2.resize(eye_img, (self.eye_width, self.eye_height), interpolation=cv2.INTER_CUBIC)

        # pupil_detection algorithm
        #cx, cy, pupil_img = self.pupil_detection(img, fps)
        cx, cy, pupil_img = 0,0,eye_img
        
        pupil_cx = int(round(self.cx*480))
        pupil_cy = int(round(self.cy*360))
        
        # pupil dist from eye's rigid point
        # self.anchor_eye_x = abs(((self.cx*w) + eye_sx) - ref_lmx)
        # self.anchor_eye_y = ((self.cy*h) + eye_sy) - ref_lmy
        
        self.anchor_eye_x = 0
        self.anchor_eye_y = 0

        #if cx > 0 and cy > 0:
        #    # Use absolute pupil coordinates in the camera frame
        #    self.eye_x = (self.cx*w) + eye_sx
        #    self.eye_y = (self.cy*h) + eye_sy
            
        self.eye_x=x_rot #얼굴 수평방향
        self.eye_y=y_rot #얼굴 수직방향
            
        # visualize sr, pupil detection process
        vis_image = np.ones((self.disp_height, self.disp_width, 3), dtype=np.uint8)*30
        vis_image[:margin_sy,:] = (55,55,55)
        vis_image[margin_ey:,:] = (55,55,55)
        vis_image[:,:margin_sx] = (55,55,55)
        vis_image[:,margin_ex:] = (55,55,55)
        vis_image = cv2.resize(vis_image, (self.disp_width, self.disp_height), interpolation=cv2.INTER_CUBIC)

        # visualize sr, pupil detection process
        if display:
            temp = cv2.resize(pupil_img, (480, 360), interpolation=cv2.INTER_CUBIC)
            
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_CUBIC)
            
            if pupil_cx > 0 and pupil_cy > 0:
                cv2.line(img, (pupil_cx-5, pupil_cy), (pupil_cx+5, pupil_cy), (0,0,255), 2)
                cv2.line(img, (pupil_cx, pupil_cy-5), (pupil_cx, pupil_cy+5), (0,0,255), 2)
            temp_color = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
            
            combined_img = np.hstack((img, temp_color))
            cv2.imshow("eye_image_combined", combined_img)
            
        ###################################### Calibration proceess ######################################
        if self.timer < self.time_step/2.0:
            self.mode = "calibration"
            
            if self.timer > -self.time_step/2.0:
                self.draw_cali_circle(vis_image, tuple(np.array([int(0.5*(margin_ex - margin_sx) + margin_sx), \
                    int(0.5*(margin_ey - margin_sy) + margin_sy)])), 21, line_size, self.timer, -self.time_step/2.0)
            if self.timer > -self.time_step/2.0 + int(self.time_step/2):
                self.eye_cali0.append([self.anchor_eye_x, self.anchor_eye_y])
                
            if display:
                cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Calibration', vis_image)
        ###################################### 1st calibration point #####################################        
        elif self.timer < self.time_step*2:
            self.mode = "calibration"
            if self.timer > self.time_step:
                self.draw_cali_circle(vis_image, tuple(np.array([int(0.1*(margin_ex - margin_sx) + margin_sx), \
                    int(0.1*(margin_ey - margin_sy) + margin_sy)])), 21, line_size, self.timer, self.time_step)
            if self.timer > self.time_step + int(self.time_step/2):
                self.eye_cali1.append([self.eye_x, self.eye_y])
            
            if display:
                cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Calibration', vis_image)
        ###################################### 2nd calibration point #####################################        
        elif self.timer < self.time_step*3:
            self.mode = "calibration"
            if self.timer > self.time_step*2:
                self.draw_cali_circle(vis_image, tuple(np.array([int(0.9*(margin_ex - margin_sx) + margin_sx), \
                    int(0.1*(margin_ey - margin_sy) + margin_sy)])), 21, line_size, self.timer, self.time_step*2)
            if self.timer > self.time_step*2 + int(self.time_step/2):
                self.eye_cali2.append([self.eye_x, self.eye_y])
            
            if display:  
                cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Calibration', vis_image)
        ###################################### 3rd calibration point #####################################        
        elif self.timer < self.time_step*4:
            self.mode = "calibration"
            if self.timer > self.time_step*3:
                self.draw_cali_circle(vis_image, tuple(np.array([int(0.9*(margin_ex - margin_sx) + margin_sx), \
                    int(0.9*(margin_ey - margin_sy) + margin_sy)])), 21, line_size, self.timer, self.time_step*3)
            if self.timer > self.time_step*3 + int(self.time_step/2):
                self.eye_cali3.append([self.eye_x, self.eye_y])
            
            if display:
                cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Calibration', vis_image)
        ###################################### 4th calibration point #####################################        
        elif self.timer < self.time_step*5:
            self.mode = "calibration"
            if self.timer > self.time_step*4:
                self.draw_cali_circle(vis_image, tuple(np.array([int(0.1*(margin_ex - margin_sx) + margin_sx), \
                    int(0.9*(margin_ey - margin_sy) + margin_sy)])), 21, line_size, self.timer, self.time_step*4)
            if self.timer > self.time_step*4 + int(self.time_step/2):
                self.eye_cali4.append([self.eye_x, self.eye_y])
            
            if display:
                cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Calibration', vis_image)
            
        else:
            self.mode = "tracking"
            if len(self.eye_cali1) > 0:
        ###################################### Get calibration matrix #####################################        
                if len(self.eye_cali1) < 2 or len(self.eye_cali2) < 2 or len(self.eye_cali3) < 2 or len(self.eye_cali4) < 2:
                    raise Exception("Calibration Err, reference frame shortage")
                
                self.eye0 = np.mean(self.eye_cali0[len(self.eye_cali0)//2:], axis=0) if len(self.eye_cali0)>1 else [0,0]
                eye1 = np.mean(self.eye_cali1[len(self.eye_cali1)//2:], axis=0)
                eye2 = np.mean(self.eye_cali2[len(self.eye_cali2)//2:], axis=0)
                eye3 = np.mean(self.eye_cali3[len(self.eye_cali3)//2:], axis=0)
                eye4 = np.mean(self.eye_cali4[len(self.eye_cali4)//2:], axis=0)
                eye_pt1 = np.float32([[eye1[0],eye1[1]], [eye2[0],eye2[1]], 
                    [eye4[0],eye4[1]], [eye3[0],eye3[1]]])
                eye_pt2 = np.float32([[0.1,0.1],[0.9,0.1],[0.1,0.9],[0.9,0.9]])
                
                # calc perspective matrix
                self.eye_PM = cv2.getPerspectiveTransform(eye_pt1, eye_pt2)
                # Message (calibrateion success)
                
                self.eye_cali1 = []
                self.eye_cali2 = []
                self.eye_cali3 = []
                self.eye_cali4 = []
                
            try:
                cv2.destroyWindow('Calibration')
            except:
                pass
            
            self.eye_xdeg.append(self.eye_x)
            self.eye_ydeg.append(self.eye_y)
        
            if len(self.eye_xdeg)>=fps/2:
                while len(self.eye_xdeg)>fps/2:
                    self.eye_xdeg.pop(0)
            if len(self.eye_ydeg)>=fps/2:
                while len(self.eye_ydeg)>fps/2:
                    self.eye_ydeg.pop(0)
                    
    def get_gazedata(self, fps, gaze_mode):
        gaze_x, gaze_y, diff_cx, diff_cy = -1, -1, -1, -1
        
        if self.eye_PM is not None:
            try:
                eye_x = np.mean(self.eye_xdeg)
                eye_y = np.mean(self.eye_ydeg)
                eye_pts = cv2.perspectiveTransform(np.array([((eye_x, eye_y),)]), self.eye_PM)

                gaze_x = int(eye_pts[0][0][0]*self.disp_width)
                gaze_y = int(eye_pts[0][0][1]*self.disp_height)
                
                self.gaze_xs.append(gaze_x)
                self.gaze_ys.append(gaze_y)
            
                if len(self.gaze_xs)>=fps/2:
                    while len(self.gaze_xs)>fps/2:
                        self.gaze_xs.pop(0)
                if len(self.gaze_ys)>=fps/2:
                    while len(self.gaze_ys)>fps/2:
                        self.gaze_ys.pop(0)
                
                if gaze_mode == "combine":
                    pupil_cx = self.eye0[0]
                    pupil_cy = self.eye0[1]
                    
                    # 1포인트 보정과의 차이값에 상수곱하는 연산(scaling)
                    diff_cx = (self.anchor_eye_x - pupil_cx)*10
                    diff_cy = (self.anchor_eye_y - pupil_cy)*10
                    
                    # 가중치 설정
                    wx = 0.1
                    wy = 0.1
                    
                    # Head gaze 및 eye difference 값의 가중치합
                    gaze_x = int(round(np.mean(self.gaze_xs)*(1.0 - wx)) + round(diff_cx*wx))
                    gaze_y = int(round(np.mean(self.gaze_ys)*(1.0 - wy)) + round(diff_cy*wy))
                elif gaze_mode == "head" or gaze_mode == "eye":
                    gaze_x = int(round(np.mean(self.gaze_xs)))
                    gaze_y = int(round(np.mean(self.gaze_ys)))
                else:
                    pass
                
            except Exception as e:
                print(str(e))
        else:
            return -1, -1, -1, -1
        return gaze_x, gaze_y, diff_cx, diff_cy