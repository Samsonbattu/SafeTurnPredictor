import cv2
from road_segmenter_DeepLabv3 import segment_road

cap = cv2.VideoCapture("datasets/videos/test_video1.mp4")

ret, frame = cap.read()

road_mask = segment_road(frame)

print("Frame shape:", frame.shape)
print("Mask shape:", road_mask.shape)

cv2.imwrite("mask_output.png", road_mask)

print("Mask saved successfully")

cap.release()
cv2.destroyAllWindows()