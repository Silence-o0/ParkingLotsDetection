import argparse
import sys

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from clustering import assign_clusters
from inference import inference
from prep_processing import make_alter_spots

import threading
from queue import Queue

def process_image(model, conf, file_path, output_path="result.png", plot=False):
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
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()


def process_video(model, conf, video_source):
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(str(video_source))

    if not cap.isOpened():
        raise ValueError(f"File can't be opened: {video_source}")

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 500, 300)

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
            cv2.imshow('Video', vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        input_queue.put(None)
        thread.join()
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Detect parking spots from stream or file.")
    parser.add_argument('--source', type=str, required=True,
                        help='Use "stream" for live video or provide path to image/video file.')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save the output (only for image input).')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Optional. Show the result visually (only for image input). Default is False.')
    args = parser.parse_args()

    model = YOLO("model.pt")
    conf = 0.5

    if args.source == "stream":
        process_video(model, conf, 0)
    else:
        print(f"Processing file: {args.source}")
        try:
            if args.source.lower().endswith(('.png', '.jpg', '.jpeg')):
                if args.output is not None:
                    process_image(model, conf, args.source, args.output, plot=args.plot)
                else:
                    process_image(model, conf, args.source, plot=args.plot)
            else:
                process_video(model, conf, args.source)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
