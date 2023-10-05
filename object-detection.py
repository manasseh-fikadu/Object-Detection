import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()


@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(
        pretrained=True, pretrained_backbone=True, weights=weights, box_score_tresh=0.8)
    model.eval()
    return model


model = load_model()


def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[i] for i in prediction["labels"]]
    return prediction


def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_boxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"], colors=[
                                         "red" if label == "person" else "green" for label in prediction["labels"]], width=2)
    img_with_boxes_np = img_with_boxes.detach().numpy().transpose(1, 2, 0)
    return img_with_boxes_np

# Dashboard
st.title("Object Detector")
upload = st.file_uploader(label="Upload Image here: ", type=["png", "jpg", "jpeg"])

if upload is not None:
    image = Image.open(upload)
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = make_prediction(image)
    image_width_bbox = create_image_with_bboxes(np.array(image).transpose(), prediction)

    st.header("Image with Bounding Boxes")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.imshow(image_width_bbox)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    st.header("Prediction Probabilities")
    st.write(prediction)
