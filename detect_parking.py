# Provide video/image processing pipeline
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from clustering import assign_clusters
from inference import inference
from preprocessing import make_alter_spots

import threading
from queue import Queue


def process_image(model, conf, file_path, output_path="result.png", plot=False):
    """Processes an image to detect and visualize parking spots.

    Args:
        model (YOLO): Weight of the model
        conf (float): Confidence threshold for detection (0-1)
        file_path (str): Path to input image file
        output_path (str, optional): Path to save output image. Defaults to "result.png"
        plot (bool, optional): Whether to display the result. Defaults to False
    """
    image = cv2.imread(file_path)
    spots = inference(model, conf, image)

    overlay = image.copy()
    alpha = 0.3

    alter_spots, _ = make_alter_spots(spots, image)
    assign_clusters(alter_spots, tolerance_coef=0.3)

    for spot in spots:
        spot.draw_classes(image, overlay, alpha)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    plt.figure(figsize=(16, 16))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    if plot:
        plt.show()


def process_video(model, conf, video_source):
    """Processes video stream or file to detect parking spots in real-time and visualize it.

    Args:
        model (YOLO): Weight of the model
        conf (float): Confidence threshold for detection (0-1)
        video_source (str/int): Video file path or camera index (0 for default camera)
    """
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"File can't be opened: {video_source}")
    else:
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            raise ValueError(f"File can't be opened: {video_source}")

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 500, 300)

    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)
    last_spots = []

    def inference_worker():
        while True:
            frame = input_queue.get()
            if frame is None:
                break

            spots = inference(model, conf=conf, image=frame)
            alter_spots, _ = make_alter_spots(spots, frame)
            assign_clusters(alter_spots, tolerance_coef=0.3)
            output_queue.put(spots)

    thread = threading.Thread(target=inference_worker)
    thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if input_queue.empty():
                input_queue.put(frame.copy())

            if not output_queue.empty():
                last_spots = output_queue.get()

            overlay = frame.copy()
            for spot in last_spots:
                spot.draw_classes(frame, overlay, alpha=0.3)

            vis = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.imshow("Video", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        input_queue.put(None)
        thread.join()
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Handles command line arguments and routes processing to appropriate functions.
    Supports both image files and video streams.

    Command Line Args:
        --source: 'stream' for live video or path to image/video file
        --output: Optional output path (image processing only)
        --plot: Whether to display results (image processing only)
    """
    parser = argparse.ArgumentParser(
        description="Detect parking spots from stream or file."
    )
    parser.add_argument(
        "source",
        type=str,
        help='Camera index (0,1,2,...) or path to image/video file.'
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output (only for image input).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optional. Show the result visually (only for image input).",
    )
    args = parser.parse_args()

    model = YOLO("model.pt")
    conf = 0.5

    try:
        video_source = int(args.source)
        process_video(model, conf, video_source)
    except ValueError:
        video_source = args.source
        if video_source.lower().endswith((".png", ".jpg", ".jpeg")):
            if args.output is not None:
                process_image(model, conf, video_source, args.output, plot=args.plot)
            else:
                process_image(model, conf, video_source, plot=args.plot)
        else:
            process_video(model, conf, video_source)


if __name__ == "__main__":
    main()
