import cv2
from  road_segmenter_DeepLabv3 import segment_road
cap = cv2.VideoCapture(
"datasets/videos/test_video1.mp4"
)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    road_mask = segment_road(frame)
    cv2.imshow("original",frame)
    cv2.imshow("road mask",road_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    print("Frame:", frame_count)

    road_mask = segment_road(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Road Mask", road_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break