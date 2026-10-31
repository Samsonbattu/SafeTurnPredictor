import cv2

from road_mask_extractor import extract_road_mask
from road_boundary_detector import detect_boundaries

video_path = "your_video.mp4"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # temporary frame
    cv2.imwrite("temp_frame.jpg", frame)

    mask = extract_road_mask("temp_frame.jpg")

    left_points, right_points = detect_boundaries(mask)

    result = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    for x, y in left_points:
        cv2.circle(
            result,
            (x, y),
            3,
            (255, 0, 0),
            -1
        )

    for x, y in right_points:
        cv2.circle(
            result,
            (x, y),
            3,
            (0, 0, 255),
            -1
        )

    cv2.imshow("Road Mask", mask)
    cv2.imshow("Road Boundaries", result)

    key = cv2.waitKey(30)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()