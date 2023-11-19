import numpy as np
import cv2

left = cv2.VideoCapture(0)

# # Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
# left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Grab both frames first, then retrieve to minimize latency between cameras
while left.grab():
    _, leftFrame = left.retrieve()
    leftWidth, leftHeight = leftFrame.shape[:2]

    cv2.imshow('left', leftFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
cv2.destroyAllWindows()