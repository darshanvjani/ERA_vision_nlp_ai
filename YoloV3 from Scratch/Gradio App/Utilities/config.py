import os

import torch

MAIN_DIR = "/kaggle/working/"
# DATASET = os.path.join(MAIN_DIR, "../data/PASCAL_VOC")
DATASET = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 2
BATCH_SIZE = 40
IMAGE_SIZE = 416
INPUT_RESOLUTIONS = [416, 544]
INPUT_RESOLUTIONS_CUM_PROBS = [50, 100]
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_PATH = os.path.join(MAIN_DIR, "Store/checkpoints/")
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
TRAIN_MOSAIC_PERCENTAGE = 0.5
TEST_MOSAIC_PERCENTAGE = 0.00
MODEL_STATE_DICT_PATH = os.path.join(MAIN_DIR, "Store/checkpoints/yolov3.pth")

MODEL_CHECKPOINT_PATH = "./Store/epoch=39-step=16560.ckpt"
EXAMPLE_IMAGE_PATH = "./Store/examples/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

means = [0.485, 0.456, 0.406]

scale = 1.1

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]