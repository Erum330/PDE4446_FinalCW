import cv2
import serial
import time

# Adjust COM port: Windows: "COM3", Linux: "/dev/ttyUSB0"
ser = serial.Serial('COM3', 115200)
time.sleep(2)

cap = cv2.VideoCapture(0)

pan = 0.0
tilt = 0.0
step = 0.05

def send_servo_values(pan, tilt):
    msg = f"{pan} {tilt}\n"
    ser.write(msg.encode('utf-8'))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera View", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    if key == ord('a'):  # left
        pan = max(-1.0, pan - step)
    if key == ord('d'):  # right
        pan = min(1.0, pan + step)
    if key == ord('w'):  # up
        tilt = min(1.0, tilt + step)
    if key == ord('s'):  # down
        tilt = max(-1.0, tilt - step)

    send_servo_values(pan, tilt)

cap.release()
cv2.destroyAllWindows()
