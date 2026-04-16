import cv2

# 🔁 set these based on your ffmpeg output
CAM0_INDEX = 1
CAM1_INDEX = 2

cap0 = cv2.VideoCapture(CAM0_INDEX, cv2.CAP_AVFOUNDATION)
cap1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_AVFOUNDATION)

if not cap0.isOpened():
    raise RuntimeError(f"Camera {CAM0_INDEX} failed to open")
if not cap1.isOpened():
    raise RuntimeError(f"Camera {CAM1_INDEX} failed to open")

ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

cap0.release()
cap1.release()

if not ret0:
    raise RuntimeError("Failed to read from camera 0")
if not ret1:
    raise RuntimeError("Failed to read from camera 1")

# save images
cv2.imwrite("cam0.jpg", frame0)
cv2.imwrite("cam1.jpg", frame1)

print("Saved cam0.jpg and cam1.jpg")