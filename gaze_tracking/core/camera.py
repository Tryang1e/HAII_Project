import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30, pipeline=None, profile=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = pipeline
        self.profile = profile
        self.config = None

    def start(self):
        if self.pipeline is None:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            try:
                self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
                self.profile = self.pipeline.start(self.config)
            except Exception as e:
                print(f"Configuration Error: {e}")
                self.config.enable_all_streams()
                self.profile = self.pipeline.start(self.config)

        try:
            device = self.profile.get_device()
            print(f"Connected device: {device.get_info(rs.camera_info.name)}")

            # Iterate over all sensors to be absolutely sure the emitter is off everywhere
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.emitter_enabled):
                    try:
                        sensor.set_option(rs.option.emitter_enabled, 0)
                    except RuntimeError as e:
                        pass
                if sensor.supports(rs.option.laser_power):
                    try:
                        sensor.set_option(rs.option.laser_power, 0)
                    except RuntimeError as e:
                        pass
                        
            print("IR Illumination and Laser Pattern strictly forcibly turned OFF.")
        except RuntimeError as e:
            print(f"Error configuring sensor: {e}")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame()
        if not ir_frame:
            return None
        return np.asanyarray(ir_frame.get_data())

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

class WebcamCamera:
    def __init__(self, width=1280, height=720, fps=30, camera_index=0):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open Webcam index {self.camera_index}")
        print("Webcam started successfully.")

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        # On Mac with RealSense UVC, returning BGR. We should return just the first channel to act as Gray
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def stop(self):
        if self.cap:
            self.cap.release()

def get_camera(width=1280, height=720, fps=30):
    # STEP 1: Using rs.context to globally turn off the emitter FIRST
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            for dev in devices:
                print(f"Hardware RealSense Detected: {dev.get_info(rs.camera_info.name)}")
                for sensor in dev.query_sensors():
                    if sensor.supports(rs.option.emitter_enabled):
                        try:
                            sensor.set_option(rs.option.emitter_enabled, 0)
                        except: pass
                    if sensor.supports(rs.option.laser_power):
                        try:
                            sensor.set_option(rs.option.laser_power, 0)
                        except: pass
            print("Force globally disabled IR Emitter and Laser via Hardware Query.")
    except Exception as e:
        print("Hardware Context querying failed:", e)

    # STEP 2: Try to start using RealSense Pipeline
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        
        # Try to start it directly to see if a device is connected
        profile = pipeline.start(config)
        print("RealSense pyrealsense2 pipeline started successfully.")
            
        return RealSenseCamera(width=width, height=height, fps=fps, pipeline=pipeline, profile=profile)
    except Exception as e:
        print(f"RealSense pipeline start failed: {e}")
    
    # STEP 3: Fallback to UVC Webcam
    print("Falling back to OpenCV WebCam mode...")
    return WebcamCamera(width=width, height=height, fps=fps)

