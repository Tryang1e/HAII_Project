from time import perf_counter
from time import sleep


class Timer:
    # FPS
    video_fps = 30
    bio_fps = 30
    eyegaze_fps = 30
    gesture_fps = 30

    # Time stamps
    video_time_stamps = []
    bio_time_stamps = []
    eyegaze_time_stamps = []
    gesture_time_stamps = []
    window_size = 30

    # Timer
    BIO_BPM_PERIOD = 5  # sec
    bio_timer_t = 0

    @classmethod
    def set_video_time_stamp(cls):
        cls.video_time_stamps.append(perf_counter())
        if len(cls.video_time_stamps) > cls.window_size:
            del cls.video_time_stamps[0]
        cls.video_fps = 30 if len(cls.video_time_stamps) == 1 else (len(cls.video_time_stamps) - 1) / (cls.video_time_stamps[-1] - cls.video_time_stamps[0])

    @classmethod
    def set_bio_time_stamp(cls):
        cls.bio_time_stamps.append(perf_counter())
        if len(cls.bio_time_stamps) > cls.window_size:
            del cls.bio_time_stamps[0]
        cls.bio_fps = 30 if len(cls.bio_time_stamps) == 1 else (len(cls.bio_time_stamps) - 1) / (cls.bio_time_stamps[-1] - cls.bio_time_stamps[0])

    @classmethod
    def set_eyegaze_time_stamp(cls):
        cls.eyegaze_time_stamps.append(perf_counter())
        if len(cls.eyegaze_time_stamps) > cls.window_size:
            del cls.eyegaze_time_stamps[0]
        cls.eyegaze_fps = 30 if len(cls.eyegaze_time_stamps) == 1 else (len(cls.eyegaze_time_stamps) - 1) / (cls.eyegaze_time_stamps[-1] - cls.eyegaze_time_stamps[0])

    @classmethod
    def set_gesture_time_stamp(cls):
        cls.gesture_time_stamps.append(perf_counter())
        if len(cls.gesture_time_stamps) > cls.window_size:
            del cls.gesture_time_stamps[0]
        cls.gesture_fps = 30 if len(cls.gesture_time_stamps) == 1 else (len(cls.gesture_time_stamps) - 1) / (cls.gesture_time_stamps[-1] - cls.gesture_time_stamps[0])

    @classmethod
    def get_video_fps(cls):
        return cls.video_fps

    @classmethod
    def get_bio_fps(cls):
        return cls.bio_fps

    @classmethod
    def get_eyegaze_fps(cls):
        return cls.eyegaze_fps

    @classmethod
    def get_gesture_fps(cls):
        return cls.gesture_fps

    @classmethod
    def check_bio_timer(cls):
        curr_t = perf_counter()

        if cls.bio_timer_t == 0:
            cls.bio_timer_t = curr_t
            return True
        elif (curr_t - cls.bio_timer_t) > cls.BIO_BPM_PERIOD:
            cls.bio_timer_t = curr_t
            return True
        else:
            return False

    @classmethod
    def sleep(cls, t):
        sleep(t)
