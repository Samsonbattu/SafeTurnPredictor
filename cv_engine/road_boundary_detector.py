import cv2
import numpy as np

def detect_boundaries(mask):
    height, width = mask.shape
    left_points = []
    right_points = []
    for y in range(height):
        row = mask[y]
        white_pixels = np.where(row>0)[0]
        if len(white_pixels)>0:
            left_points.append(
                (white_pixels[0],y)
            )
            right_points.append(
                (white_pixels[-1],y)
            )
    return left_points, right_points