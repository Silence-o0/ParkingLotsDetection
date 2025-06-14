# Main class ParkingSpot

import random
import cv2


class ParkingSpot:
    """Core class that represent detected parking spots."""
    def __init__(self, cls, conf, center, contour):
        self.cls = cls  # Class
        self.conf = conf  # Confidence
        self.center = center  # (cx, cy)
        self.contour = contour  # Contour (np.array shape (N,1,2))
        self.cluster = -1  # Cluster index
        self.area = self.compute_area()  # Area

    def draw_classes(self, image, overlay, alpha=0.2):
        if self.cluster != -1:
            if self.cls == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.fillPoly(overlay, [self.contour], color=color)
            cv2.polylines(image, [self.contour], isClosed=True, color=color, thickness=2)
            cv2.circle(
                image,
                self.center,
                radius=3,
                color=(255, 255, 255),
                thickness=3,
            )

    def __repr__(self):
        return (
            f"ParkingSpot(cls={self.cls}, conf={self.conf:.2f}, "
            f"center={self.center}, cluster={self.cluster}, "
            f"contour_points={len(self.contour)},"
            f"area={self.area:.2f})"
        )

    def compute_area(self):
        return cv2.contourArea(self.contour)

    def get_cluster_color(self):
        random.seed(self.cluster)
        return tuple(random.randint(50, 255) for _ in range(3))

    def visualize_cluster(self, image, overlay, index=None):
        # color = (0, 0, 0) if getattr(self, 'cluster', -1) == -1 else self.get_cluster_color()
        if self.cluster != -1:
            color = self.get_cluster_color()

            if self.contour is not None:
                cv2.fillPoly(overlay, [self.contour], color=color)
                cv2.polylines(
                    image,
                    [self.contour],
                    isClosed=True,
                    color=color,
                    thickness=2,
                )
                cv2.circle(
                    image,
                    tuple(map(int, self.center)),
                    radius=4,
                    color=(255, 255, 255),
                    thickness=3,
                )

            if index is not None:
                text_pos = (int(self.center[0] + 5), int(self.center[1] - 5))
                text_color = (
                    (255, 255, 255) if getattr(self, "cluster", -1) == -1 else (0, 0, 0)
                )
                cv2.putText(
                    image,
                    str(index),
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=text_color,
                    thickness=12,
                    lineType=cv2.LINE_AA,
                )
