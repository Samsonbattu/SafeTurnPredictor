import cv2
import numpy as np


def extract_road_mask(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # Convert image to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Detect green drivable area from YOLOP output
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    road_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Larger kernel to reconnect regions
    kernel = np.ones((15, 15), np.uint8)

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Debug view
    #cv2.imshow("Original", image)
    #cv2.imshow("Mask", road_mask)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return road_mask