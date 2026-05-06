import cv2
import time
import csv
import os
from datetime import datetime

from core.camera import get_camera
from detector.landmark import Detector

# normal, one, two, swap
# ================= Configuration =================
GESTURE_PREFIX = "quit"  # name of the gesture class
RECORD_DURATION = 1.0   # fix 1.0 seconds duration
# =================================================

def main():
    print(f"Starting Data Logger. Target Gesture: {GESTURE_PREFIX}, Duration: {RECORD_DURATION}s")
    
    # Initialize camera and detector
    camera = get_camera(width=1280, height=720, fps=60)
    detector = Detector()
    
    camera.start()
    
    is_recording = False
    start_time = None
    gesture_data = []  # To store the time-series angle data
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
                
            # Convert frame for display
            if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                display_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                display_image = frame.copy()
            
            h_img, w_img = display_image.shape[:2]
            
            # 1. Process Hand Tracker
            has_hands = detector.process_hands(display_image)
            
            # Simple drawing for visualization
            if has_hands and detector.hand_results.multi_hand_landmarks:
                import mediapipe.python.solutions.drawing_utils as mp_drawing
                import mediapipe.python.solutions.hands as mp_hands
                for hand_landmarks in detector.hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 2. Extract Data & Recording Logic
            if is_recording:
                elapsed = time.time() - start_time
                if elapsed <= RECORD_DURATION:
                    # Retrieve the mapped joint angles
                    angles = detector.get_joint_angles()
                    
                    if angles is not None:
                        # row: [elapsed_time, angle_0, angle_1, ... angle_29]
                        row = [round(elapsed, 4)] + angles
                        gesture_data.append(row)
                    
                    # Draw visual progress bar
                    progress_ratio = min(elapsed / RECORD_DURATION, 1.0)
                    bar_w = int(progress_ratio * w_img)
                    cv2.rectangle(display_image, (0, h_img - 30), (bar_w, h_img), (0, 0, 255), -1)
                    
                    cv2.putText(display_image, f"Recording: {elapsed:.1f}s / {RECORD_DURATION:.1f}s", 
                                (10, h_img - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Reached duration, stop and save
                    is_recording = False
                    if gesture_data:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{GESTURE_PREFIX}_data_{timestamp}.csv"
                        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gesture_dataset')
                        out_path = os.path.join(dataset_dir, filename)
                        
                        with open(out_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            # Write headers
                            headers = ["time_sec"] + [f"angle_{i}" for i in range(30)]
                            writer.writerow(headers)
                            writer.writerows(gesture_data)
                        print(f"[{filename}] Saved {len(gesture_data)} frames of angle data.")
                    else:
                        print("No valid hands detected during recording. Discarding.")
                    gesture_data = [] # Reset for next recording
            else:
                # Idle state
                cv2.putText(display_image, "Press 'r' to Start Recording. 'q' to Quit.", (10, h_img - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 3. Permanent Info overlay
            info_text = f"Class: {GESTURE_PREFIX} | Duration: {RECORD_DURATION}s"
            font_scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x, text_y = 10, 30
            padding = 5
            
            rect_x1 = text_x - padding
            rect_y1 = text_y - text_size[1] - padding
            rect_x2 = text_x + text_size[0] + padding
            rect_y2 = text_y + padding
            
            cv2.rectangle(display_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            cv2.putText(display_image, info_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

            # View Image
            output_image = cv2.resize(display_image, (w_img // 2, h_img // 2))
            cv2.imshow('Gesture Data Collector', output_image)
            
            # Key Bindings
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and not is_recording:
                #time.sleep(2)
                is_recording = True
                start_time = time.time()
                gesture_data = []
                print(f"Rec started: {GESTURE_PREFIX}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
