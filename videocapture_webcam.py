import cv2 as cv
import numpy as np

# --- Configuration for Red HSV ---
# These values define the range of "red" color.
# We use two ranges since red wraps around the Hue (H) spectrum.
LOWER_RED_1 = np.array([0, 120, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 120, 100])
UPPER_RED_2 = np.array([180, 255, 255])

def initial_mask():
    capture = cv.VideoCapture(0)
    if not capture.isOpened(): return

    while True:
        isTrue, frame = capture.read()
        cv.imshow("Video", frame)

        hsv_frame=cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --- MASKING CODE ---
        # 1. Create two masks
        mask1 = cv.inRange(hsv_frame, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv.inRange(hsv_frame, LOWER_RED_2, UPPER_RED_2)
        
        # 2. Combine them
        initial_mask = cv.bitwise_or(mask1, mask2)
        
        cv.imshow('Original Frame', frame)
        cv.imshow('Initial RAW Red Mask', initial_mask)


        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    initial_mask()