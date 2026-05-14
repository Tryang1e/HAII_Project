import cv2

class UIManager:
    @staticmethod
    def draw_landmarks(display_image, detector, w_img, h_img):
        if detector.hand_results and detector.hand_results.multi_hand_landmarks:
            for hand_landmarks in detector.hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                    cv2.circle(display_image, (cx, cy), 5, (0, 255, 255), -1)

    @staticmethod
    def draw_zone(display_image, zx_sx, zy_sy, zx_ex, zy_ey):
        cv2.rectangle(display_image, (zx_sx, zy_sy), (zx_ex, zy_ey), (0, 0, 255), 3)
        cv2.putText(display_image, "MODE", (zx_sx + 10, zy_sy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    @staticmethod
    def draw_progress(display_image, zx_sx, zy_sy, zx_ex, zy_ey, progress_h):
        cv2.rectangle(display_image, (zx_sx, zy_ey - progress_h), (zx_ex, zy_ey), (0, 0, 255), -1)

    @staticmethod
    def draw_hand_position(display_image, hx, hy, w_img, h_img):
        cv2.circle(display_image, (int(hx * w_img), int(hy * h_img)), 10, (255, 255, 0), -1)

    @staticmethod
    def draw_virtual_touch_mode(display_image, roi_x_min, roi_y_min, roi_x_max, roi_y_max, w_img, h_img):
        cv2.putText(display_image, "VIRTUAL TOUCH MODE", (w_img//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(display_image, (int(roi_x_min * w_img), int(roi_y_min * h_img)), 
                      (int(roi_x_max * w_img), int(roi_y_max * h_img)), (255, 255, 0), 1)

    @staticmethod
    def draw_hand_tracking_mode(display_image, w_img):
        cv2.putText(display_image, "HAND TRACKING MODE", (w_img//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
