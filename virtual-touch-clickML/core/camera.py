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
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.profile = self.pipeline.start(self.config)
            except Exception as e:
                print(f"Configuration Error: {e}")
                self.config.enable_all_streams()
                self.profile = self.pipeline.start(self.config)

        try:
            device = self.profile.get_device()
            print(f"Connected device: {device.get_info(rs.camera_info.name)}")

            sensor = device.query_sensors()[0]
            if sensor.supports(rs.option.emitter_enabled):
                try:
                    sensor.set_option(rs.option.emitter_enabled, 2)
                    print("IR Illumination turned ON.")
                except RuntimeError:
                    sensor.set_option(rs.option.emitter_enabled, 0)
        except RuntimeError as e:
            print(f"Error configuring sensor: {e}")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

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
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
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
        return frame

    def stop(self):
        if self.cap:
            self.cap.release()

def get_camera(width=1280, height=720, fps=30):
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Try to start it directly to see if a device is connected
        profile = pipeline.start(config)
        print("RealSense device detected.")
        return RealSenseCamera(width=width, height=height, fps=fps, pipeline=pipeline, profile=profile)
    except Exception as e:
        print(f"RealSense check failed: {e}")
    
    print("RealSense not found, falling back to WebCam...")
    return WebcamCamera(width=width, height=height, fps=fps)

