from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import cv2
import numpy as np


def point_in_polygons(points, poligons):
    ponto = Point(points)

    for poligono in poligons:
        polygon = Polygon(poligono)
        if polygon.contains(ponto):
            return True
    return False


def draw_roi(frame, polygons):
    frame_overlay = frame.copy()
    for polygon in polygons:
        cv2.fillPoly(
            frame_overlay,
            np.array([polygon], dtype=np.int32),
            (0, 127, 127)
        )

    alpha = 0.3
    output_frame = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
    return output_frame
