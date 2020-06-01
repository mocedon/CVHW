#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code is the Visualizing CNN Filters segment from the tutorial appendix,
with minor changes.
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
import seaborn as sns
import torch.nn as nn

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
def plot_filters_single_channel_big(t):
    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    #     fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)

    return fig


def plot_filters_single_channel(t):
    # kernels depth * number of kernels
    nplots = t.shape[0] * t.shape[1]
    ncols = 12

    nrows = 1 + nplots // ncols

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()

    return fig


def plot_filters_multi_channel(t):
    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    #     plt.savefig('myimage.png', dpi=100)
    plt.tight_layout()

    return fig


def plot_weights(model, layer_num, single_channel=True, collated=False, outname=None, n_filters=None):
    # extracting the model features at the particular layer number
    layer = model.features[layer_num]

    fig = None
    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.features[layer_num].weight.data[slice(n_filters)]

        if single_channel:
            if collated:
                fig = plot_filters_single_channel_big(weight_tensor)
            else:
                fig = plot_filters_single_channel(weight_tensor)
            return fig
        else:
            if weight_tensor.shape[1] == 3:
                fig = plot_filters_multi_channel(weight_tensor)
            else:
                print("Can only plot weights with three channels with single channel = False")
    else:
        print("Can only visualize layers which are convolutional")

    # Save figure if desired
    try:
        fig.savefig((OUTPUT_FOLDER / outname).with_suffix(".png"),
                    bbox_inches="tight", pad_inches=0)
    except (UnboundLocalError, TypeError):
        print("Figure not saved")

    return fig
