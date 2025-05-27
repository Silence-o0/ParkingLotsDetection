# Parking Spots Detection

A system for detecting vacant and occuied parking spots using YOLOv11 segmentation and clustering.

![Example image](example_result1.png)
![Example image](example_result2.png)

## Dataset

- **Source:** [Dataset Link](https://www.kaggle.com/datasets/silenceo0/parkinglotsaugmented)
- **Size:** 1,270 annotated images for training and 298 for validation
- **Format:** YOLO segmentation (`class x1 y1 x2 y2 ...`)
- **Classes:** Vacant, Occupied
- **Augmentations:**
  - Horizontal flip
  - Rotating
  - Brightness adjustment
  - Scaling
  - Cropping

## Model

YOLO11m-seg (for instance segmentation) was trained on the custom dataset of parking spots.

### Training Details

- **Epochs:** 100  
- **Batch Size:** 8  
- **Image Size:** 960  
- Other hyperparameters were kept at their default values.

## Clustering

A custom clustering method groups parking spots into blocks. Clusters with only one or two spots are considered noise and ignored. The algorithm use BFS to find nearby spots without a fixed distance parameter. Distances depend on the size of the spots, which helps handle different spacing caused by camera perspective.

## Installation

```bash
git clone https://github.com/Silence-o0/ParkingLotsDetection
cd ParkingLotsDetection
pip install -r requirements.txt
````

## Usage

### Detect from webcam (live stream)

```bash
python detect_parking.py <camera_id>
```

### Detect from an image or videofile

```bash
python detect_parking.py <path/to/file>
```

### Flags

* `--output`

  * Optional. Path to save the result (only for image input).

* `--plot`

  * Optional. Show the result visually (only for image input).

