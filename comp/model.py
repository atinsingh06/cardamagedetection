# %%
#pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
#pip install git+https://github.com/facebookresearch/detectron2.git@4a5e6d79e626837a0317195131afaca64b3f4e2d

#python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html

# %%
#%matplotlib inline
from pycocotools.coco import COCO
import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
pylab.rcParams['figure.figsize'] = (8.0, 10.0)# Import Libraries
import cv2
# For visualization
import os
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
#import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Set base params
#plt.rcParams["figure.figsize"] = [16,9]
#!python -m detectron2.utils.collect_env

# %%
# I am visualizing some images in the 'val/' directory

dataDir=r'C:\Users\atin3\Downloads\archive (3)\val'
dataType='COCO_val_annos'
mul_dataType='COCO_mul_val_annos'
annFile='{}/{}.json'.format(dataDir,dataType)
mul_annFile='{}/{}.json'.format(dataDir,mul_dataType)
img_dir = r"C:\Users\atin3\Downloads\archive (3)\img"
# initialize coco api for instance annotations
coco=COCO(annFile)
mul_coco=COCO(mul_annFile)


dataset_dir = r"C:\Users\atin3\Downloads\archive (3)"
img_dir = "img/"
train_dir = "train/"
val_dir = "val/"
from detectron2.data.datasets import register_coco_instances
register_coco_instances("car_dataset_train", {}, os.path.join(dataset_dir,train_dir,"COCO_train_annos.json"), os.path.join(dataset_dir,img_dir))
register_coco_instances("car_dataset_val", {}, os.path.join(dataset_dir,val_dir,"COCO_val_annos.json"), os.path.join(dataset_dir,img_dir))

# %%
import distutils.version
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("car_dataset_train",)
cfg.DATASETS.TEST = ("car_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 0#increase for speed only when using gpu
cfg.MODEL.WEIGHTS =r"C:\Users\atin3\Downloads\data-20240521T054143Z-001\data\output\model_final.pth" # Let training initialize from model zoo
cfg.MODEL.DEVICE = 'cpu'

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001 # pick a good LR
cfg.SOLVER.WARMUP_ITERS = 800
cfg.SOLVER.MAX_ITER = 1600 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (600, 1550)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # faster, and good enough for this dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 600



# Clear any logs from previous runs
#TODO add timestamp to logs
cfg.OUTPUT_DIR=r"C:\Users\atin3\Downloads\data-20240521T054143Z-001\data\output"


#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
#trainer.train()
predictor = DefaultPredictor(cfg)

# %%
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image

# %%
def is_covering(box1, box2):
    """
    Check if box1 completely covers box2.

    Args:
        box1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        box2 (tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
        bool: True if box1 completely covers box2, False otherwise.
    """
    return box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3]

def remove_covering_boxes(boxes):
    """
    Remove bounding boxes that are completely covered by other boxes.

    Args:
        boxes (list): List of bounding boxes, where each box is represented as a tuple (x1, y1, x2, y2).

    Returns:
        list: List of non-covering bounding boxes.
    """
    # Initialize list to store non-covering boxes
    non_covering_boxes = []

    # Iterate through each pair of boxes
    for i, box1 in enumerate(boxes):
        is_covered = False
        for j, box2 in enumerate(boxes):
            if i != j:
                # Check if box1 completely covers box2
                if is_covering(box1, box2):
                    is_covered = True
                    break
        # If box1 is not completely covered by any other box, add it to the list of non-covering boxes
        if not is_covered:
            non_covering_boxes.append(box1)

    return non_covering_boxes




# %%
def generate_bbox(path):
    image = cv2.imread(path)
    #plt.imshow(image)
    #plt.show()

    height, width, channels = image.shape

    # Print the image size
    print(f"Image size: {width}x{height}")

    # Resize the image
    #image = cv2.resize(image, (1024, 1024))
    

    outputs = predictor(image)
    #print(outputs)
    instances = outputs["instances"]
    #print(instances.pred_boxes.tensor.cpu().numpy())
    #print(outputs)
    # Retrieve confidence scores for predicted bounding boxes
    confidence_scores = instances.scores.tolist()

    # Set a threshold for confidence scores
    confidence_threshold = 0.75  # Adjust this threshold as needed

    # Filter instances based on confidence threshold
    selected_indices = [i for i, score in enumerate(confidence_scores) if score >= confidence_threshold]
    filtered_instances = instances[selected_indices]
    # Get the predicted bounding boxes
    pred_bboxes = filtered_instances.pred_boxes.tensor.cpu().numpy()
    confidence_scores = filtered_instances.scores.tolist()
    # Print the predicted bounding boxes
    #for bbox in pred_bboxes:
    pred_bboxes = remove_covering_boxes(pred_bboxes)
    #print(list(pred_bboxes))

    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN), scale=1.2)
    #out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    #plt.imshow(out.get_image()[:, :, ::-1])
    #plt.show()
    #print(confidence_scores)

    for rectangle in pred_bboxes:
        x1, y1, x2, y2 = map(int,rectangle)
        cv2.rectangle(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # Display the image with rectangles
    import matplotlib.pyplot as plt
    return image


# %%

# %%

# %%



