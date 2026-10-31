import cv2

from road_mask_extractor import extract_road_mask
from road_boundary_detector import detect_boundaries

path = "cv_engine/segmentation/YOLOP_experiment/YOLOP/inference/output/test1.jpg"

mask = extract_road_mask(path)
cv2.imwrite("road_mask_debug.png", mask)
print("Road mask saved as road_mask_debug.png")

left_points, right_points = detect_boundaries(mask)

result = cv2.cvtColor(
    mask,
    cv2.COLOR_GRAY2BGR
)

for x, y in left_points:
    cv2.circle(
        result,
        (x, y),
        4,
        (0, 0, 255),
        -1
    )

for x, y in right_points:
    cv2.circle(
        result,
        (x, y),
        4,
        (255, 0, 0),
        -1
    )

cv2.imshow("Boundaries", result)

cv2.waitKey(0)

cv2.destroyAllWindows()