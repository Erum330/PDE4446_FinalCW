import cv2 as cv
import numpy as np

# --- Configuration for Red HSV ---
# These values define the range of "red" color.
# We use two ranges since red wraps around the Hue (H) spectrum.
LOWER_RED_1 = np.array([0, 150, 120])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 150, 120])
UPPER_RED_2 = np.array([180, 255, 255])

def robust_mask():
    capture = cv.VideoCapture(0)
    if not capture.isOpened(): return

    # Define the kernel (a small matrix) used for cleaning
    kernel = np.ones((7,7), np.uint8)

    while True:
        isTrue, frame = capture.read()

        hsv_frame=cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --- MASKING CODE ---
        # 1. Create two masks
        mask1 = cv.inRange(hsv_frame, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv.inRange(hsv_frame, LOWER_RED_2, UPPER_RED_2)
        
        # 2. Combine them
        initial_mask = cv.bitwise_or(mask1, mask2)

       # This is equivalent to DILATE then ERODE
        cleaned_mask = cv.morphologyEx(initial_mask, cv.MORPH_CLOSE, kernel)
        
        # 2. Opening: Removes small white specks outside the main shape (removes background noise)
        cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_OPEN, kernel)
        
        cv.imshow('Original Frame', frame)
        cv.imshow('Cleaned Red Mask', cleaned_mask)



        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    robust_mask()