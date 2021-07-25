from fastapi import FastAPI
from pydantic import BaseModel

# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")   # please manually install torch 1.8 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image
import numpy as np
import io, json
import base64

# -*- coding: utf-8 -*- 
import re
import base64
from io import BytesIO

# pip3 install pillow
from PIL import Image

app = FastAPI()


# define the Input class
class Input(BaseModel):
    base64str : str

def base64str_to_PILImage(base64str):
	base64_img_bytes = base64str.encode('utf-8')
	base64bytes = base64.b64decode(base64_img_bytes)
	bytesObj = io.BytesIO(base64bytes)
	img = Image.open(bytesObj) 
	return img

def PILImage_to_cv2(img):
	return np.asarray(img)


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

@app.put("/predict/")
def get_predictionbase64(d:Input):
    '''
    FastAPI API will take a base 64 image as input and return a json object
    '''
    # Load the image
    img = base64str_to_PILImage(d.base64str)
    img = PILImage_to_cv2(img)
    outputs = predictor(img)

    return {'boxes': outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist(),
        'classes' : outputs["instances"].pred_classes.cpu().numpy().tolist()}