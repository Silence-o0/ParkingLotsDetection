# Inference and save results as ParkingSpot objects

import numpy as np
import cv2

from parking_spot import ParkingSpot


def save_to_class_array(masks, classes, confs, H, W):
    """Converts model outputs into ParkingSpot objects by processing masks and contours.

    Args:
        masks (np.ndarray): Array of segmentation masks from model prediction
        classes (np.ndarray): Array of class IDs for each detected object
        confs (np.ndarray): Array of confidence scores for each detection
        H (int): Original image height
        W (int): Original image width

    Returns:
        list[ParkingSpot]: List of validated parking spot objects with:
            - Class ID
            - Confidence score
            - Center coordinates
            - Contour points
    """
    spots = []

    for i, cls in enumerate(classes):
        conf = confs[i]
        best_mask = masks[i].astype(np.uint8) * 255
        best_mask = cv2.resize(best_mask, (W, H))

        contours, _ = cv2.findContours(
            best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) >= 4:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center = (cx, cy)

                        spot = ParkingSpot(
                            cls=cls, conf=conf, center=center, contour=approx
                        )
                        spots.append(spot)
    return spots


def inference(model, conf, image):
    """Model inference on input image.

    Args:
        model (YOLO): Weight of the model
        conf (float): Minimum confidence threshold (0-1)
        image (np.ndarray): Input image in BGR format
    """
    results = model.predict(
        image,
        imgsz=960,
        conf=conf,
        show=False,
        line_width=2,
        show_labels=False,
        show_conf=False,
    )

    if len(results) == 0 or results[0].masks is None:
        return []

    result = results[0]

    try:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        H, W = image.shape[:2]
        return save_to_class_array(masks, classes, confs, H, W)
    except Exception as e:
        print(f"Inference error: {str(e)}")
        return []
