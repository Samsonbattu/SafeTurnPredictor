import cv2
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolov8n-seg.pt")

# Load video
cap = cv2.VideoCapture("datasets/videos/test_video1.mp4")

ret, frame = cap.read()

if not ret:
    print("Could not read frame")
    exit()

# Run segmentation
results = model(frame)

# Draw predictions
#annotated_frame = results[0].plot()

# Save image
#cv2.imwrite("yolo_output.png", annotated_frame)

#print("Saved as yolo_output.png")
print(results[0].names)
cap.release()