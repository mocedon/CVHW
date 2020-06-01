#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code analyzes a pre-trained CNN (the VGG16) and utilizes it to classify
cat and dog images via Transfer Learning (TL).
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
import cv2
import skimage
from sklearn.svm import LinearSVC
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from run_as_script import list_files_in_subfolders
import visualizing_cnn_filters as vcf
import visualizing_layer_output as vlo

################################################################################
# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
################################################################################
# Files and paths constants
CODE_FOLDER = pathlib.Path(r"")
BIRDS_FOLDER = pathlib.Path(r"birds")
CATS_FOLDER = pathlib.Path(r"cats")
DOGS_FOLDER = pathlib.Path(r"dogs")
MYDATA_FOLDER = pathlib.Path(r"my_data")
OUTPUT_FOLDER = pathlib.Path(r"output\Part2")

# ImageNet-1000 class dictionary
with open("ImageNet_Classes.txt") as _:
    CLASS = eval(_.read())

# Our class dictionary
CAT = 0
DOG = 1

###############################################################################
# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
###############################################################################


###############################################################################
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
###############################################################################
# # # # # # # # # # # # # # conversion functions # # # # # # # # # # # # # # #
def pillow2numpy(im):
    """
    Cast Image.Image type to numpy.ndarray type, of numpy.float32.

    :param im: Image of type Image.Image

    :return: Image of type numpy.ndarray
    """
    im = skimage.img_as_float32(np.asarray(im))

    return im


# # # # # # # # # # # # # Module-specific functions # # # # # # # # # # # # # #
def load_image(filename: (pathlib.Path, str)) -> Image.Image:
    """
    Load an image using PIL.Image loader.

    :param filename: Path to image file

    :return: PIL.Image instance
    """
    im = Image.open(filename)

    return im


def display_image(im: (np.ndarray, torch.Tensor),
                  outname: (pathlib.Path, str) = None):
    """
    Display image using plt.imshow, and save figure if out-name is given.

    :param im: Image to display
    :param outname: Out-name to save the figure, default: None (doesn't save)
    """
    # Prepare image to display
    if isinstance(im, np.ndarray):
        # If numpy.ndarray, do nothing
        pass
    elif isinstance(im, torch.Tensor):
        # Else if torch.Tensor, convert to RGB numpy array
        im = im.numpy().squeeze().transpose((1, 2, 0))
    elif isinstance(im, Image.Image):
        # Else if PIL.Image.Image, convert to RGB numpy array
        im = np.asarray(im)
    else:
        # Else: Unsupported image type, print error message
        print("Can't display, unsupported image type")

    # Prepare to normalize image to display range: [0, 1]
    immin = im.min(axis=(0, 1), initial=+np.inf)
    immax = im.max(axis=(0, 1), initial=-np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        im_normalized = (im - immin) / (immax - immin)

    # Create figure and show normalized image
    fig, ax = plt.subplots()
    ax.imshow(im_normalized)
    ax.axis('off')
    ax.set_title(outname)
    plt.show()

    # Save figure if desired
    if outname is not None:
        fig.savefig((OUTPUT_FOLDER / outname).with_suffix(".jpg"),
                    bbox_inches="tight", pad_inches=0)


def normalize_vgg16(im: torch.Tensor) -> torch.Tensor:
    """
    Normalize batch according to ImageNet's mean and std values

    :param im: Image to normalize

    :return: Normalized image
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return normalize(im)


def preprocess_vgg16(im: (np.ndarray, torch.Tensor)) -> torch.Tensor:
    """
    Pre-process single image before fed to a net. Also send to device.

    :param im: Image to pre-process

    :return: Pre-processed image
    """
    # Convert input_ image to Image.Image unless already
    try:
        # If numpy array, convert to PIL.Image
        im = Image.fromarray(skimage.img_as_ubyte(im))
    except TypeError:
        # Else already Image.Image, do nothing
        pass

    # Pre-process procedure:
    # (1) resize to 224x224
    # (2) convert to Tensor
    # (3) normalize
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_vgg16
    ])
    # (4) Create mini-batch and send to device
    return preprocess(im).unsqueeze(0).to(device)


def predict(im, model, labels: tuple = None):
    """
    Predict image class using the given model.

    :param im: Image to classify
    :param model: Model to use for classification
    :param labels: Class names dictionary

    :return: Predicted class index, and label if given
    """
    with torch.no_grad():
        # Feed image to model, and choose category of maximal score
        _, index = model(im).data[0].max(0)
        # Return predicted class index, and label if given
        if labels is not None:
            return index.item(), labels[index.item()]
        else:
            return index.item()


def full_cycle(*, im=None, model=None, labels: tuple = None,
               inname: (pathlib.Path, str) = None,
               outname: (pathlib.Path, str) = None) -> tuple:
    """
    Predict image class using the given model.

    :param im: Image to classify. If not given, image must be provided by inname
    :param model: Model to use for classification
    :param labels: Class-names mapping
    :param inname: In-name to load the image from, default: None (doesn't load)
    :param outname: Out-name to save the figure, default: None (doesn't save)

    :return: Tuple of (1) Loaded image, (2) Prediction
    """
    # If image not explicitly given, load image
    if im is None:
        im = load_image(inname)

    # Display image
    display_image(im, outname)
    im_pp = preprocess_vgg16(im)
    display_image(im_pp, f"{outname} pre-processed")
    with torch.no_grad():
        out = predict(im_pp, model, labels=labels)
        print(f"{outname} --> {out}")

    return im, out


def rotate(im, angle, scaling=1):
    """
    Display matched points
    # Inputs          Description
    # --------------------------------------------------------------------------
    # im              Instance of the first image, saving load operation
    # angle           Rotation angle in degrees
    # scaling         Scaling factor. Default is 1.
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # im_rot          Instance of rotated image
    """
    # Rotation matrix
    h, w, _ = im.shape
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scaling)
    # Affine transformation
    im_rot = cv2.warpAffine(im, rot_mat, (w, h))

    return im_rot


###############################################################################
# -----------------------------------------------------------------------------
# When runs as a script
# -----------------------------------------------------------------------------
###############################################################################
if __name__ == "__main__":
    # 2.0) Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2.1) Load a pre-trained VGG16 in evaluation mode
    model = models.vgg16(pretrained=True, progress=True).to(device)
    model.eval()

    # 2.2) Load and display birds images
    # Get image paths
    flist = [*list_files_in_subfolders(BIRDS_FOLDER, "*")]
    # Load and display images
    birds = []
    for fname in flist:
        bird = load_image(fname)
        display_image(bird, fname.stem)
        birds.append(bird)

    # 2.3) Pre-process images to fit VGG16: 1x3x224x224 float
    birds_pp = []
    for bird, fname in zip(birds, flist):
        bird_pp = preprocess_vgg16(bird)
        display_image(bird_pp, f"{fname.stem} pre-processed")
        birds_pp.append(bird_pp)

    # 2.4) Feed the image to the model
    out_birds = []
    with torch.no_grad():
        for bird_pp, fname in zip(birds_pp, flist):
            out_bird = predict(bird_pp, model, labels=CLASS)
            print(f"{fname} --> {out_bird}")
            out_birds.append(out_bird)

    # 2.5) Feed the web-image of a cat to the model
    fname = [*list_files_in_subfolders(MYDATA_FOLDER, "Part2_Q5*.jpg")].pop()
    im_web, im_web_out = full_cycle(model=model, labels=CLASS,
                                    inname=fname, outname="Cat from-web")

    # 2.6+2.7) Apply transformations on the web-image, and feed each to the model
    im_web_np = pillow2numpy(im_web)
    # (1) Rotation: rotate image
    rotation_angle = 135
    im_web_rot, im_web_rot_out = \
        full_cycle(im=rotate(im_web_np, rotation_angle),
                   model=model, labels=CLASS,
                   inname=fname, outname="Cat from-web rotated")
    # (2) Threshold: apply threshold to image to create an RGB binary image
    threshold_rgb = np.full((1, 1, 3), 0.5, dtype=np.float32)
    im_web_thresh, im_web_thresh_out = \
        full_cycle(im=(im_web_np > threshold_rgb).astype(np.float32),
                   model=model, labels=CLASS,
                   inname=fname, outname="Cat from-web thresholded")
    # (3) Filter: apply Gaussian blurring filter
    sigma = 9
    ksize = int(2 * np.floor(3 * sigma) + 1)
    im_web_blur, im_web_blur_out = \
        full_cycle(im=cv2.GaussianBlur(im_web_np, (ksize, ksize), sigma),
                   model=model, labels=CLASS,
                   inname=fname, outname="Cat from-web blurred")

    # 2.8) Plot 3 filters of the first layer of VGG16, and their response to our images
    # (0) Three filters
    with torch.no_grad():
        layer_ = 0
        single_channel_ = False
        collated_ = False
        n_filters_ = 3
        vcf.plot_weights(model, layer_, single_channel=single_channel_, collated=collated_,
                         n_filters=n_filters_,
                         outname=f"{n_filters_} first filters of VGG16, layer={layer_},"
                                 f" single_channel={single_channel_}, collated={collated_} - ")
        # (0) Original: Source filter responses
        vlo.filter_outputs(preprocess_vgg16(im_web), model, layer_, n_filters_,
                           outname=f"First filters response to Cat from-web, "
                                   f"layer={layer_}, {n_filters_} filters")
        # (1) Rotation: Rotated filter responses
        vlo.filter_outputs(preprocess_vgg16(im_web_rot), model, layer_, n_filters_,
                           outname=f"First filters response to Cat from-web rotated, "
                                   f"layer={layer_}, {n_filters_} filters")
        # (2) Threshold: Thresholded filter responses
        vlo.filter_outputs(preprocess_vgg16(im_web_thresh), model, layer_, n_filters_,
                           outname=f"First filters response to Cat from-web thresholded, "
                                   f"layer={layer_}, {n_filters_} filters")
        # (3a) Filter: Gaussian blurred filter responses
        vlo.filter_outputs(preprocess_vgg16(im_web_blur), model, layer_, n_filters_,
                           outname=f"First filters response to Cat from-web blurred, "
                                   f"layer={layer_}, {n_filters_} filters")

    # 2.9) Load cats and dogs images and construct a feature vector (of FC-layer outputs)
    with torch.no_grad():
        layer_ = 3
        # Get cats and dogs images paths
        flist_cats = [*list_files_in_subfolders(CATS_FOLDER, "*")]
        flist_dogs = [*list_files_in_subfolders(DOGS_FOLDER, "*")]
        # For each category, iterate over each file-list:
        feat_vecs = []
        classes = []
        for c, flist in zip((CAT, DOG), (flist_cats, flist_dogs)):
            # For each file in file-list: load image, pre-process, feed into the
            #   net and return feature vector of some non-finite FC layer
            for fname in flist:
                image = load_image(fname)
                image_pp = preprocess_vgg16(image)
                vec = vlo.fc_layer_outputs(image_pp, model, layer_)
                feat_vecs.append(vec)
                classes.append(c)
        feat_ndarray = torch.cat(feat_vecs).detach().numpy()
        classes = np.asarray(classes)
        print(f"FC layer {layer_//3} has feature space of size {feat_vecs[0].size()}")

    # 2.10) Use the untrained model to extract features upon which an SVC can be trained
    # Training
    # Define training data
    x_train = feat_ndarray
    y_train = classes
    # Initialize LinearSVC classifier
    clf = LinearSVC()
    clf.fit(x_train, y_train)

    # Testing: further predicting
    with torch.no_grad():
        # Get cats and dogs images paths
        flist_cats = [*list_files_in_subfolders(MYDATA_FOLDER, "Part2_Q10_Cat*.jpg")]
        flist_dogs = [*list_files_in_subfolders(MYDATA_FOLDER, "Part2_Q10_Dog*.jpg")]
        # For each category, iterate over each file-list:
        feat_vecs = []
        classes = []
        for c, flist in zip((CAT, DOG), (flist_cats, flist_dogs)):
            # For each file in file-list: load image, pre-process, feed into the
            #   net and return feature vector of some non-finite FC layer
            for fname in flist:
                image = load_image(fname)
                image_pp = preprocess_vgg16(image)
                vec = vlo.fc_layer_outputs(image_pp, model, layer_)
                feat_vecs.append(vec)
                classes.append(c)
        feat_ndarray = torch.cat(feat_vecs).detach().numpy()
        classes = np.asarray(classes)

        # Define test data
        x_test = feat_ndarray
        y_test_groundtruth = classes

        # Compute prediction
        y_test_prediction = clf.predict(x_test)

        # Calculate success ratio
        print(f"Groundtruth: {y_test_groundtruth}")
        print(f"Prediction: {y_test_prediction}")
        successes = np.count_nonzero(y_test_groundtruth == y_test_prediction)
        total = y_test_prediction.size
        print(f"Success ratio over training set: {successes / total: %}")

    print("All Done!")
