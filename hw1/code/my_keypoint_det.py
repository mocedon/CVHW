#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code implements a Feature Detector, using a simplified version of the
    Difference of Gaussian detector (DoG).
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
from skimage import img_as_float32
import cv2
from run_as_script import RunAsScriptStore, list_files_in_subfolders

################################################################################
# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
################################################################################
# Files and paths constants
CODE_FOLDER = pathlib.Path(r"")
DATA_FOLDER = pathlib.Path(r"data")
OUTPUT_FOLDER = pathlib.Path(r"output\Part1")
MAT_NAME = pathlib.Path(r"testPattern.mat")
MAT_FNAME = CODE_FOLDER / MAT_NAME

# GaussianPyramid constants
SIGMA0 = 1
K = np.sqrt(2)
LEVELS = [-1, 0, 1, 2, 3, 4]
TH_CONTRAST = 0.03
TH_R = 12


###############################################################################
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
###############################################################################
# # # # # # # # # # # # # # conversion functions # # # # # # # # # # # # # # #
def bgr2gray(im):
    """Convert BGR to grayscale IF not grayscale, ELSE do nothing"""
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if (im.ndim > 2) else im


def bgr2rgb(im):
    """Convert BGR to RGB IF not grayscale, ELSE do nothing"""
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if (im.ndim > 2) else im


def im2double(im):
    """Convert image to Double IF not converted already, ELSE do nothing"""
    return img_as_float32(im)


def dlevels2levels(dlevels, base_level):
    """Convert DoG Pyramid levels array to Gaussian Pyramid levels array"""
    # Restore base level
    return np.array([base_level, *dlevels])


def levels2dlevels(levels):
    """Convert Gaussian Pyramid levels array to DoG Pyramid levels array"""
    # Cut base level
    return levels[1:]


def indices2keypoints(indices, levels=None):
    """Convert Pyramid entries (lvl_idx, row, col) to keypoints (x, y, lvl)"""
    # Initialize keypoints array
    keypoints = np.copy(indices)

    # Convert level-index to actual level
    if levels is not None:
        keypoints[:, 0] = list(map(lambda lvl_idx: levels[lvl_idx], indices[:, 0]))

    # Swap x/col and level columns
    keypoints[:, (0, 2)] = keypoints[:, (2, 0)]

    return keypoints


def keypoints2indices(keypoints, levels=None):
    """Convert keypoints (x, y, lvl) to Pyramid entries (lvl_idx, row, col)"""
    # Initialize indices array
    indices = np.copy(keypoints)

    # Convert levels to level-index
    if levels is not None:
        levels = list(levels)
        indices[:, 2] = list(map(lambda lvl: levels.index(lvl), keypoints[:, 2]))

    # Swap x/col and level columns
    indices[:, (0, 2)] = indices[:, (2, 0)]

    return indices


# # # # # # # # # # # # # # # Display functions # # # # # # # # # # # # # # # #
def displayPyramid(pyramid):
    """Flatten pyramid into horizontally stacked images, then display"""
    fig = plt.figure(figsize=(16, 5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')
    return fig


def displayPyramidJet(pyramid):
    """Flatten pyramid into horizontally stacked images, then display in Jet"""
    fig = plt.figure(figsize=(16, 5))
    plt.imshow(np.hstack(pyramid), cmap='jet')
    plt.axis('off')
    return fig


def displayPyramidBlobs(pyramid, locsDoG, sigma0, k, levels):
    """
    Flatten pyramid into horizontally stacked images, mark blobs as points
    #     Inputs          Description
    #     --------------------------------------------------------------------------
    #     pyramid         A matrix of grayscale images of size (len(levels),imH,imW)
    #     locsDoG         N x 3 keypoint matrix of blobs centers
    #     sigma0          Scale of the 0th image pyramid.
    #     k               Pyramid Factor.
    #     levels          Levels of pyramid
    #
    #     Outputs         Description
    #     -----------------------------------------------------------------------
    #     fig             Figure handle
    """
    # Get level *indices* array and image width
    lvl_idx = keypoints2indices(locsDoG)[:, 0]
    _, _, w = pyramid.shape

    # Calculate x in flattened-pyramid image, and update in locsDoG copy
    locsDoGPyr = np.copy(locsDoG)
    locsDoGPyr[:, 0] = (lvl_idx + 1) * w + locsDoG[:, 0]

    # Plot circle-engraved pyramid
    fig = displayImageBlobs(np.hstack(pyramid), locsDoGPyr, sigma0, k)

    return fig


def displayImageBlobs(im, locsDoG, sigma0, k):
    """
    Display image, mark blobs as circles with radii correlated to their scale
    #     Inputs          Description
    #     ----------------------------------------------------------------------
    #     im              BGR or Grayscale image within range [0,1].
    #     locsDoG         N x 3 keypoint matrix of blobs centers
    #     sigma0          Scale of the 0th image pyramid.
    #     k               Pyramid Factor.
    #     levels          Levels of pyramid
    #     levels          Levels of pyramid
    #
    #     Outputs         Description
    #     -----------------------------------------------------------------------
    #     fig             Figure handle
    """
    # Generate circles
    circles = map(lambda x, y, lvl: plt.Circle((x, y), sigma0 * k ** lvl, color='r', fill=False),
                  *np.swapaxes(locsDoG, 0, 1))

    # Plot image
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.imshow(bgr2rgb(im), cmap='gray')
    plt.axis('off')

    # Plot all circles on image
    list(map(ax.add_artist, circles))

    return fig


# # # # # # # # # # # # # # # Technical functions # # # # # # # # # # # # # # #
def createGaussianPyramid(im, sigma0, k, levels):
    """Compute Gaussian Pyramid for sigma in sigma0 * k ** levels"""
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(2 * np.floor(3 * sigma_) + 1)
        blur = cv2.GaussianBlur(im, (size, size), sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)


def createDoGPyramid(GaussianPyramid, levels):
    """
    Produces DoG Pyramid
    # INPUTS
    # GaussianPyramid - size (len(levels), shape(im))matrix of the Gaussian
    #                   Pyramid created by Gaussian blurring with a different
    #                   sigma for each layer
    # levels      - the corresponding levels of the pyramid
    #
    # OUTPUTS
    # DoGPyramid  - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #               created by differencing the Gaussian Pyramid input
    """
    # Create DoG Pyramid and crop remaining levels
    DoGPyramid = GaussianPyramid[1:] - GaussianPyramid[:-1]
    DoGLevels = levels2dlevels(levels)

    return DoGPyramid, DoGLevels


def computePrincipalCurvatureChannel(DoG):
    """
    Edge Suppression in a 1-channel DoG
    #  Takes in DoG image, a 1-channel image of a DoGPyramid generated in
    #  createDoGPyramid and returns, PrincipalCurvature, a matrix of the same
    #  size where each point contains the curvature ratio R for the
    #  corresponding point in the DoG image
    #
    #  INPUTS
    #  DoG image - size shape(im) matrix of the DoG 1-channel image
    #
    #  OUTPUTS
    #  PrincipalCurvature - size shape(im) matrix where each point contains the
    #                       curvature ratio R for the corresponding point in
    #                       the DoG 1-channel image
    """
    # Compute Sobel second order derivatives
    sobel_xx = cv2.Sobel(DoG, cv2.CV_64F, 2, 0)
    sobel_xy = cv2.Sobel(DoG, cv2.CV_64F, 1, 1)
    sobel_yy = cv2.Sobel(DoG, cv2.CV_64F, 0, 2)

    # Compute Hessian trace and determinant
    tr = sobel_xx + sobel_yy
    det = (sobel_xx * sobel_yy) - (sobel_xy ** 2)

    # Add regularization factor to denominator
    det[det == 0] = np.finfo(np.float).eps

    # Compute Principal Curvature matrix
    PrincipalCurvature = (tr ** 2) / det

    return PrincipalCurvature


def computePrincipalCurvature(DoGPyramid):
    """
    Edge Suppression
    #  Takes in DoGPyramid generated in createDoGPyramid and returns
    #  PrincipalCurvature,a matrix of the same size where each point contains the
    #  curvature ratio R for the corresponding point in the DoG pyramid
    #
    #  INPUTS
    #  DoGPyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    #  OUTPUTS
    #  PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    #                       point contains the curvature ratio R for the
    #                       corresponding point in the DoG pyramid
    """
    # Compute Principal Curvature per each channel and stack
    PrincipalCurvature = map(computePrincipalCurvatureChannel, DoGPyramid)
    PrincipalCurvature = np.stack(list(PrincipalCurvature))

    return PrincipalCurvature


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r):
    """
    Returns local extrema points in both scale and space using the DoGPyramid
    #     INPUTS
    #         DoG_pyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    #         DoG_levels  - The levels of the pyramid where the blur at each level is
    #                       outputs
    #         principal_curvature - size (len(levels) - 1, imH, imW) matrix contains the
    #                       curvature ratio R
    #         th_contrast - remove any point that is a local extremum but does not have a
    #                       DoG response magnitude above this threshold
    #         th_r        - remove any edge-like points that have too large a principal
    #                       curvature ratio
    #
    #      OUTPUTS
    #         locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
    #                scale and space, and also satisfies the two thresholds.
    """
    # 1) Mask for contrast high enough
    mask_c = (abs(DoGPyramid) > th_contrast)

    # 2.1) Mask for Principal Curvature Ratio low enough
    mask_rt = (PrincipalCurvature < th_r)
    # 2.2) Mask for Principal Curvature Ratio positive - since negative stands
    #   for opposing eigenvalues signs, hence Saddle-Points -> Not extrema!
    mask_rp = (0 < PrincipalCurvature)
    # 2) Putting them together
    mask_r = (mask_rp & mask_rt)

    # 3) Mask for both thresholds of scale-space absolute value maxima
    mask_max =\
        (DoGPyramid > np.pad(DoGPyramid[1:, :, :], ((0, 1), (0, 0), (0, 0)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:-1, :, :], ((1, 0), (0, 0), (0, 0)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, 1:, :], ((0, 0), (0, 1), (0, 0)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, :-1, :], ((0, 0), (1, 0), (0, 0)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, :, 1:], ((0, 0), (0, 0), (0, 1)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, :, :-1], ((0, 0), (0, 0), (1, 0)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, 1:, 1:], ((0, 0), (0, 1), (0, 1)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, :-1, 1:], ((0, 0), (1, 0), (0, 1)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, 1:, :-1], ((0, 0), (0, 1), (1, 0)))) &\
        (DoGPyramid > np.pad(DoGPyramid[:, :-1, :-1], ((0, 0), (1, 0), (1, 0))))
    mask_min =\
        (DoGPyramid < np.pad(DoGPyramid[1:, :, :], ((0, 1), (0, 0), (0, 0)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:-1, :, :], ((1, 0), (0, 0), (0, 0)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, 1:, :], ((0, 0), (0, 1), (0, 0)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, :-1, :], ((0, 0), (1, 0), (0, 0)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, :, 1:], ((0, 0), (0, 0), (0, 1)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, :, :-1], ((0, 0), (0, 0), (1, 0)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, 1:, 1:], ((0, 0), (0, 1), (0, 1)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, :-1, 1:], ((0, 0), (1, 0), (0, 1)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, 1:, :-1], ((0, 0), (0, 1), (1, 0)))) &\
        (DoGPyramid < np.pad(DoGPyramid[:, :-1, :-1], ((0, 0), (1, 0), (1, 0))))
    mask_e = (mask_max | mask_min)

    # Locations of local maxima satisfying both thresholds.
    mask = mask_c & mask_r & mask_e
    indices = np.argwhere(mask)

    # Convert to (x,y,lvl) format
    locsDoG = indices2keypoints(indices, DoGLevels)

    return locsDoG


def DoGdetector(im, sigma0, k, levels, th_contrast, th_r):
    """
    Putting it all together
    #     Inputs          Description
    #     --------------------------------------------------------------------------
    #     im              Grayscale or color image with range [0,1].
    #     sigma0          Scale of the 0th image pyramid.
    #     k               Pyramid Factor.  Suggest sqrt(2).
    #     levels          Levels of pyramid to construct. Suggest -1:4.
    #     th_contrast     DoG contrast threshold.  Suggest 0.03.
    #     th_r            Principal Ratio threshold.  Suggest 12.
    #
    #     Outputs         Description
    #     --------------------------------------------------------------------------
    #     locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
    #                     in both scale and space, and satisfies the two thresholds.
    #     gauss_pyramid   A matrix of grayscale images of size (len(levels),imH,imW)
    """
    # If necessary, convert image to grayscale and float32 within [0,1]
    im_gs = bgr2gray(img_as_float32(im))

    # Compute Gaussian Pyramid
    GaussianPyramid = createGaussianPyramid(im_gs, sigma0, k, levels)

    # Compute DoG Pyramid
    DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)

    # Compute Principal Curvature Ratio corresponding tensor
    PrincipalCurvature = computePrincipalCurvature(DoGPyramid)

    # Find DoG keypoints
    locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r)

    return locsDoG, GaussianPyramid,


###############################################################################
# -----------------------------------------------------------------------------
# When runs as a script
# -----------------------------------------------------------------------------
###############################################################################
if __name__ == "__main__":
    # A non namespace wasting stoage for Run As Script variables
    ras = RunAsScriptStore()

    # Store general parameters in ras
    ras.sigma0 = SIGMA0
    ras.k = K
    ras.levels = LEVELS
    ras.th_contrast = TH_CONTRAST
    ras.th_r = TH_R

    # 1.1) Load and show model_chickenbroth.jpg image
    ras.im = cv2.imread(str(DATA_FOLDER / "model_chickenbroth.jpg"))
    ras.fig = plt.figure(figsize=(16, 5))
    plt.imshow(cv2.cvtColor(ras.im, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    ras.fig.savefig(OUTPUT_FOLDER / "model_chickenbroth - Original.png",
                    bbox_inches="tight", pad_inches=0)

    # 1.2) Convert image to grayscale-Double, compute Gaussian Pyramid and show
    # Convert image to grayscale and float32 within [0,1]
    ras.im_gs = bgr2gray(img_as_float32(ras.im))
    # Compute Gaussian Pyramid
    ras.GaussianPyramid = \
        createGaussianPyramid(ras.im_gs, ras.sigma0, ras.k, ras.levels)
    # Print the shape of the Gaussian Pyramid
    print(f"Shape of Gaussian Pyramid: {ras.GaussianPyramid.shape}")
    # Visualize the Gaussian Pyramid
    ras.fig = displayPyramid(ras.GaussianPyramid)
    ras.fig.savefig(OUTPUT_FOLDER / "model_chickenbroth - GaussianPyramid.png",
                    bbox_inches="tight", pad_inches=0)

    # 1.3) Compute Difference of Gaussian Pyramid and show
    # Compute Difference of Gaussian Pyramid
    ras.DoGPyramid, ras.DoGLevels = \
        createDoGPyramid(ras.GaussianPyramid, ras.levels)
    # Print the shape of the Gaussian Pyramid
    print(f"Shape of DoG Pyramid: {ras.DoGPyramid.shape}")
    # Visualize the Gaussian Pyramid
    ras.fig = displayPyramid(ras.DoGPyramid)
    ras.fig.savefig(OUTPUT_FOLDER / "model_chickenbroth - DoGPyramid.png",
                    bbox_inches="tight", pad_inches=0)

    # 1.4) Compute Principal Curvatures of the Difference of Gaussian Pyramid
    ras.PrincipalCurvature = computePrincipalCurvature(ras.DoGPyramid)

    # 1.5) Compute Principal Curvatures of the Difference of Gaussian Pyramid
    ras.locsDoG = \
        getLocalExtrema(ras.DoGPyramid, ras.DoGLevels, ras.PrincipalCurvature,
                        ras.th_contrast, ras.th_r)
    print("Keypoints detected (x, y, level):")
    print(ras.locsDoG)

    # 1.6) Run DoG detector end-to-end and display notated images
    # a) Apply to model_chickenbroth.jpg
    # b) Apply to incline_R.png
    # c) Apply to self-taken Panoramic_Desert*.jpg
    ras.flist = \
        {*list_files_in_subfolders(DATA_FOLDER, "*model_chickenbroth*"),
         *list_files_in_subfolders(DATA_FOLDER, "*incline_R*"),
         *list_files_in_subfolders(DATA_FOLDER, "*Desert*"),
         # *list_files_in_subfolders(DATA_FOLDER, "*Gonen20_L*"),
         # *list_files_in_subfolders(DATA_FOLDER, "*Dead-Sea*"),
         # *list_files_in_subfolders(DATA_FOLDER, "*Lake*"),
         # *list_files_in_subfolders(DATA_FOLDER, "*Plant*"),
         # *list_files_in_subfolders(DATA_FOLDER, "*"),
         }
    for ras.i, ras.fname in enumerate(ras.flist):
        ras.im = cv2.imread(str(ras.fname))
        ras.locsDoG, ras.GaussianPyramid = \
            DoGdetector(ras.im, ras.sigma0, ras.k, ras.levels, ras.th_contrast, ras.th_r)
        ras.fig = displayImageBlobs(ras.im, ras.locsDoG, ras.sigma0, ras.k)
        ras.fig.savefig(OUTPUT_FOLDER / f"{ras.fname.stem} - Features Detected on Image.png",
                        bbox_inches="tight", pad_inches=0)
        plt.clf()
        ras.fig = displayPyramidBlobs(ras.GaussianPyramid, ras.locsDoG, ras.sigma0, ras.k, ras.levels)
        ras.fig.savefig(OUTPUT_FOLDER / f"{ras.fname.stem} - Features Detected on Gaussian Pyramid.png",
                        bbox_inches="tight", pad_inches=0)
        plt.clf()
        print(f"{ras.i + 1}/{len(ras.flist)} Done!")
    print("All Done!")
