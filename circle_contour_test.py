import cv2 as cv
import numpy as np

# --- Configuration ---
LOWER_RED_1 = np.array([0, 150, 120])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 150, 120])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 500
WIDTH, HEIGHT = 640, 480
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# --- Angle Calculation Function (REMOVED) ---
# The function and control variables are removed.

def run_tracking_and_vision():
    
    capture = cv.VideoCapture(1)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not capture.isOpened(): 
        print("Error: Could not open webcam.")
        return

    print("Vision Tracking Active (Contour, Centroid, and Pixel Error). Press 'd' to exit.")

    # Define the kernel used for cleaning
    kernel = np.ones((7,7), np.uint8)

    while True:
        isTrue, frame = capture.read()
        if not isTrue: break

        hsv_frame=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_display = hsv_frame.copy() 

        # Masking and Robust morphology
        mask1 = cv.inRange(hsv_frame, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv.inRange(hsv_frame, LOWER_RED_2, UPPER_RED_2)
        initial_mask = cv.bitwise_or(mask1, mask2)
        cleaned_mask = cv.morphologyEx(initial_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_OPEN, kernel)
        
        # Contour Finding
        contours, _ = cv.findContours(cleaned_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Draw the target center (white crosshair)
        cv.line(frame, (CENTER_X - 15, CENTER_Y), (CENTER_X + 15, CENTER_Y), (255, 255, 255), 2)
        cv.line(frame, (CENTER_X, CENTER_Y - 15), (CENTER_X, CENTER_Y + 15), (255, 255, 255), 2)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv.contourArea)
            
            if cv.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                
                # --- CIRCULAR BOUNDING BOX & CENTROID ---
                
                # Use minEnclosingCircle to get the bounding circle parameters
                ((x, y), radius) = cv.minEnclosingCircle(largest_contour)
                center_of_circle = (int(x), int(y))
                
                # *** ADJUSTED LOGIC: Shrink the radius for better visual fit ***
                visual_radius = int(radius * 0.70) # Reduce radius by 30% for visual effect

                # Draw the bounding circle (Yellow)
                contour_color = (0, 255, 255) 
                cv.circle(frame, center_of_circle, visual_radius, contour_color, 2)
                cv.circle(hsv_display, center_of_circle, visual_radius, contour_color, 2)

                # Get the true center of mass (Centroid) for the aiming point
                M = cv.moments(largest_contour)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    # Draw object center (green crosshair) - This is the actual aim point
                    cv.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
                    cv.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
                    
                    # --- Calculate Pixel Error ---
                    error_x = center_x - CENTER_X
                    error_y = center_y - CENTER_Y
                    
                    # Display Results
                    cv.putText(frame, f"X ERROR: {error_x} px", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.putText(frame, f"Y ERROR: {error_y} px", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv.imshow('1. Original Frame (Tracking Error)', frame)
        cv.imshow('2. HSV Frame (Bounding Circle)', hsv_display)
        cv.imshow('3. Cleaned Binary Mask', cleaned_mask)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    run_tracking_and_vision()