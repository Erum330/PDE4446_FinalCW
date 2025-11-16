import cv2 as cv

capture=cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    cv.imshow("Video", frame)

    hsv_frame=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow("HSV_Video", hsv_frame)


    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)