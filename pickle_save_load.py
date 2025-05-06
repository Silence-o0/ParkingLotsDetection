# Save to pickle, load from pickle

from ultralytics import YOLO
import cv2
import pickle

from inference import inference


def save_pickle():
    model = YOLO("data/best_cm30.pt")
    image_path = "data/test1.JPG"
    spots_pickle_path = "data/mspots1.pkl"
    conf = 0.5

    spots = inference(model, conf, image_path)

    with open(spots_pickle_path, 'wb') as f:
        pickle.dump(spots, f)

def load_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        spots = pickle.load(f)
    return spots

save_pickle()