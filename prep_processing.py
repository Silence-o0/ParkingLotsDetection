# Make quadrangles

import cv2
import numpy as np


class ParkingSpotAlter:
    def __init__(self, ps, w, h, w_k, w_b, h_k, h_b):
        self.ps = ps
        self.width = w
        self.height = h
        self.w_line = (w_k, w_b)
        self.h_line = (h_k, h_b)


def max_area_quad(contour):
    max_area = 0
    best_quad = None

    n = len(contour)
    for p1 in range(n - 3):
        for p2 in range(p1 + 1, n - 2):
             for p3 in range(p2 + 1, n - 1):
                 for p4 in range(p3 + 1, n):
                    quad = [contour[p1], contour[p2], contour[p3], contour[p4]]
                    quad = np.array(quad, dtype=np.float32)
                    area = cv2.contourArea(quad)
                    if area > max_area:
                       max_area = area
                       best_quad = quad
    return best_quad

def segment_length(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def line_params(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        k = float('inf')
        b = x1
    else:
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
    return k, b

def average_line_params(p1, p2, p3, p4):
    k1, b1 = line_params(p1, p2)
    k2, b2 = line_params(p3, p4)

    if k1 == float('inf') and k2 == float('inf'):
        return float('inf'), (b1 + b2) / 2
    elif k1 == float('inf') or k2 == float('inf'):
        return float('inf'), b1 if k1 == float('inf') else b2
    else:
        return (k1 + k2) / 2, (b1 + b2) / 2


def convert_with_geometry(spots):
    altered_spots = []
    for ps in spots:
        contour = ps.contour.squeeze()
        if contour.shape != (4, 2):
            continue

        p0, p1, p2, p3 = contour

        width = (segment_length(p0, p1) + segment_length(p2, p3)) / 2
        w_k, w_b = average_line_params(p0, p1, p2, p3)

        height = (segment_length(p1, p2) + segment_length(p3, p0)) / 2
        h_k, h_b = average_line_params(p1, p2, p3, p0)

        altered = ParkingSpotAlter(
            ps,
            width,
            height,
            w_k,
            w_b,
            h_k,
            h_b
        )
        altered_spots.append(altered)
    return altered_spots



def draw_lines_on_image(image, altered_spots, color_w=(0, 255, 0), color_h=(255, 0, 0), thickness=10):
    img_copy = image.copy()
    height, width = img_copy.shape[:2]

    for spot in altered_spots:
        cv2.circle(img_copy, tuple(spot.ps.center), 8, (0, 0, 255), 20)
        k_w, b_w = spot.w_line
        if k_w != float('inf'):
            x0, x1 = 0, width - 1
            y0 = int(k_w * x0 + b_w)
            y1 = int(k_w * x1 + b_w)
            cv2.line(img_copy, (x0, y0), (x1, y1), color_w, thickness)
        else:
            x = int(b_w)
            cv2.line(img_copy, (x, 0), (x, height - 1), color_w, thickness)

        k_h, b_h = spot.h_line
        if k_h != float('inf'):
            x0, x1 = 0, width - 1
            y0 = int(k_h * x0 + b_h)
            y1 = int(k_h * x1 + b_h)
            cv2.line(img_copy, (x0, y0), (x1, y1), color_h, thickness)
        else:
            x = int(b_h)
            cv2.line(img_copy, (x, 0), (x, height - 1), color_h, thickness)

    return img_copy


def make_alter_spots(filter_spots, image):
    for spot in filter_spots:
        if len(spot.contour)!=4:
            contour = spot.contour.squeeze()
            quad = max_area_quad(contour)

            if quad is not None:
                spot.contour = quad.astype(int)

    alter_spots = convert_with_geometry(filter_spots)
    # image = draw_lines_on_image(image.copy(), alter_spots)
    return alter_spots


