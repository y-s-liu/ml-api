import base64
import io
import json
import random
from email import header

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_option("deprecation.showPyplotGlobalUse", False)

COCO_INSTANCE_CATEGORY_train2017 = [
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

# use file uploader object to recieve image
# Remember that this bytes object can be used only once
def bytesioObj_to_base64str(bytesObj):
    return base64.b64encode(bytesObj.read()).decode("utf-8")


# Image conversion functions


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode("utf-8")
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


def PILImage_to_cv2(img):
    return np.asarray(img)


def drawboundingbox(img, boxes, pred_cls, rect_th=2, text_size=1, text_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    # initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0]), int(boxes[i][1])),
            (int(boxes[i][2]), int(boxes[i][3])),
            color=class_color_dict[pred_cls[i]],
            thickness=rect_th,
        )
        cv2.putText(
            img,
            str(COCO_INSTANCE_CATEGORY_train2017[pred_cls[i]]),
            (int(boxes[i][0]), int(boxes[i][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            class_color_dict[pred_cls[i]],
            thickness=text_th,
        )  # Write the prediction class
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    # plt.show()


st.markdown("<h1>Our Object Detector App using FastAPI</h1><br>", unsafe_allow_html=True)

bytesObj = st.file_uploader("Choose an image file")


if bytesObj:
    # In streamlit we will get a bytesIO object from the file_uploader
    # and we convert it to base64str for our FastAPI
    if bytesObj:
        base64str = bytesioObj_to_base64str(bytesObj)

    # We will also create the image in PIL Image format using this base64 str
    # Will use this image to show in matplotlib in streamlit
    img = base64str_to_PILImage(base64str)

    # Run FastAPI
    payload = json.dumps({"base64str": base64str, "threshold": 0.5})

    response = requests.put(
        "http://mlsteam.ai.icrd/proxy/u9e39a57/predict/",
        data=payload,
        headers={"Content-Type": "application/json"},
    )  # detectron2
    # response = requests.put("http://mlsteam.ai.icrd/proxy/uab5594a/predict/",data = payload, headers={'Content-Type': 'application/json'}) # mmdetection
    data_dict = response.json()

    st.markdown("<center><h1>App Result</h1></center>", unsafe_allow_html=True)
    drawboundingbox(img, data_dict["boxes"], data_dict["classes"])
    st.pyplot()
    st.markdown("<center><h1>FastAPI Response</h1></center><br>", unsafe_allow_html=True)
    st.write(data_dict)
