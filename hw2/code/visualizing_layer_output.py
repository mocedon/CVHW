#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code is the Visualizing Layer Output segment from the tutorial appendix
"""

###############################################################################
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
###############################################################################
import pathlib
import numpy as np
import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

###############################################################################
# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
###############################################################################
# Files and paths constants
OUTPUT_FOLDER = pathlib.Path(r"output\Part2")


###############################################################################
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
###############################################################################
def to_grayscale(image):
    """
    input_ is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image


def normalize(image, device=torch.device("cpu")):
    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _normalize
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    return image


def predict(image, model, labels=None):
    _, index = model(image).data[0].max(0)
    if labels is not None:
        return str(index.item()), labels[str(index.item())][1]
    else:
        return str(index.item())


def deprocess(image, device=torch.device("cpu")):
    return image * torch.tensor([0.229, 0.224, 0.225]).to(device) + torch.tensor([0.485, 0.456, 0.406]).to(device)


def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image


def fc_layer_outputs(input_, model, layer_to_output):
    input_ = model.features(input_)
    input_ = model.avgpool(input_)
    input_ = input_.view(1, -1)

    modulelist = list(model.classifier.modules())
    output = None
    for count, layer in enumerate(modulelist[1:]):
        input_ = layer(input_)
        if count == layer_to_output:
            output = input_

    return output


def filter_outputs(image, model, layer_to_visualize, n_filters=None, outname=None):
    modulelist = list(model.features.modules())
    if layer_to_visualize < 0:
        layer_to_visualize += 31
    output = None
    for count, layer in enumerate(modulelist[1:]):
        image = layer(image)
        if count == layer_to_visualize:
            output = image

    filters = []
    output = output.data.squeeze().data[slice(n_filters)].cpu().numpy()
    for i in range(output.shape[0]):
        filters.append(output[i, :, :])

    fig = plt.figure(figsize=(10, 10))

    n = int(np.ceil(np.sqrt(len(filters))))
    m = len(filters) // n

    for i in range(len(filters)):
        ax = fig.add_subplot(m + 1, n, i + 1)
        ax.imshow(filters[i])
        ax.set_axis_off()
        # ax.set_title(i)
    plt.tight_layout()

    # Save figure if desired
    try:
        fig.savefig((OUTPUT_FOLDER / outname).with_suffix(".png"),
                    bbox_inches="tight", pad_inches=0)
    except (UnboundLocalError, TypeError, AttributeError):
        print("Figure not saved")
