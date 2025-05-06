# Inference and save results to pickle

import numpy as np
import cv2

from ParkingSpot import ParkingSpot


def save_to_class_array(masks, classes, confs, H, W):
    spots = []

    for i, cls in enumerate(classes):
        conf = confs[i]
        best_mask = masks[i].astype(np.uint8) * 255
        best_mask = cv2.resize(best_mask, (W, H))

        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) >= 4:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center = (cx, cy)

                        spot = ParkingSpot(cls=cls, conf=conf, center=center, contour=approx)
                        spots.append(spot)
    return spots


def inference(model, conf, image_path):
    results = model.predict(image_path, imgsz=960, conf=conf, show=False, line_width=2, show_labels=False, show_conf=False)

    result = results[0]
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    return save_to_class_array(masks, classes, confs, H, W)


