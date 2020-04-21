#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code implements a BRIEF Feature Descriptor.
"""

###############################################################################
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
###############################################################################
import time
import pathlib
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage
import cv2
from run_as_script import RunAsScriptStore
import my_keypoint_det as my_sift


################################################################################
# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
################################################################################

# Files and paths constants
CODE_FOLDER = pathlib.Path(r"")
DATA_FOLDER = pathlib.Path(r"..\data")
OUTPUT_FOLDER = pathlib.Path(r"..\output\Part2")
MAT_NAME = pathlib.Path(r"testPattern.mat")
MAT_FNAME = CODE_FOLDER / MAT_NAME

# GaussianPyramid constants
SIGMA0 = 1
K = np.sqrt(2)
LEVELS = [-1, 0, 1, 2, 3, 4]
TH_CONTRAST = 0.03
TH_R = 12

# BRIEF-descriptor constants
PATCH_WIDTH = 9
NBITS = 256

# Match constants
MATCH_RATIO = 0.8

# scipy.io.loadmat constants
VAR_NAMES = {"compareX": None, "compareY": None}


###############################################################################
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
###############################################################################
def mask_within_bounds(ndarray, indices, is_negative_ok):
    """
    Create mask for indices that are within bounds of ndarray
    # Inputs            Description
    # --------------------------------------------------------------------------
    # ndarray           An ndarray of any dimension and shape
    # indices           A 2d array of indices, shape: (?, ndarray.ndim)
    # is_negative_ok    When fold is True, negative indexing is valid
    #
    # Outputs           Description
    # --------------------------------------------------------------------------
    # mask              A mask with True where indices are within bounds,
    #                       shape: (len(indices), 1)
    """
    # Input validity check
    assert indices.shape[1] == ndarray.ndim, \
        "Dimension of coordinates mismatch number of matrix entries!"

    shape = np.array(ndarray.shape)
    # Mask indices below upper bound
    mask_ub = (indices < shape)
    # Mask indices above lower bound
    mask_lb = (-shape <= indices) if is_negative_ok else (0 <= indices)
    # Mask for both conditions
    mask = np.all(mask_lb, axis=1) & np.all(mask_ub, axis=1)

    return mask


def makeTestPatternI(patchWidth, nbits):
    """
    Generate a set of test pairs using sampling geometry I
    # Inputs          Description
    # --------------------------------------------------------------------------
    # patchWidth      Size of the ledge of the squared patch. Suggest 9
    # nbits           Number of comparison tests. Suggest 256
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # compareX        shape (nbits, 2) matrix of (x,y) pairs of test pixel #1
    # compareY        shape (nbits, 2) matrix of (x,y) pairs of test pixel #2
    """
    # Parameter setting
    low = -(patchWidth // 2)
    high = (patchWidth // 2)

    # Randomly generate test (x,y) locations pairs -
    #   Repeated test pairs will be regenerated!
    # Initialize array and bad values mask
    rv = np.empty((nbits, 4), dtype=int)
    mask_dup = np.ones(rv.shape, dtype=np.bool_)

    # While there are bad values, keep regenerating
    while np.any(mask_dup):
        # Regenerate new values to location of bad values
        rv[mask_dup] = \
            np.random.randint(low, high + 1, np.count_nonzero(mask_dup))

        # Mask for duplicated test pairs: unique rows SHOULD STAY UNCHANGED
        _, i_unique = np.unique(rv, axis=0, return_index=True)
        # Initialize all-are-not-unique mask.
        mask_dup = np.ones(rv.shape, dtype=np.bool_)
        # Update yes-unique places to stay unchanged
        mask_dup[i_unique] = False

    # Split into compareX, compareY
    compareX, compareY = rv[:, 0:2], rv[:, 2:4]

    return compareX, compareY


def makeTestPatternII(patchWidth, nbits):
    """
    Generate a set of test pairs using sampling geometry II
    # Inputs          Description
    # --------------------------------------------------------------------------
    # patchWidth      Size of the ledge of the squared patch. Suggest 9
    # nbits           Number of comparison tests. Suggest 256
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # compareX        shape (nbits, 2) matrix of (x,y) pairs of test pixel #1
    # compareY        shape (nbits, 2) matrix of (x,y) pairs of test pixel #2
    """
    # Parameter setting
    mu = 0
    sigma = patchWidth / 5
    low = -(patchWidth // 2)
    high = (patchWidth // 2)

    # Randomly generate test (x,y) locations pairs -
    #   Off-bounds values AND repeated test pairs will be regenerated!
    # Initialize array and bad values mask
    rv = np.empty((nbits, 4), dtype=int)
    mask_regen = np.ones(rv.shape, dtype=np.bool_)

    # While there are bad value, keep regenerating
    while np.any(mask_regen):
        # Regenerate new values to location of bad values
        rv[mask_regen] =\
            np.rint(sigma * np.random.randn(np.count_nonzero(mask_regen)) + mu)

        # Mask for off bounds values
        mask_off = (rv < low) | (rv > high)

        # Mask for duplicated test pairs: unique rows SHOULD STAY UNCHANGED
        _, i_unique = np.unique(rv, axis=0, return_index=True)
        # Initialize all-are-not-unique mask.
        mask_dup = np.ones(rv.shape, dtype=np.bool_)
        # Update yes-unique places to stay unchanged
        mask_dup[i_unique] = False

        # Combine both masks: logical OR, because fault in one gives fault
        mask_regen = mask_off | mask_dup

    # Split into compareX, compareY
    compareX, compareY = rv[:, 0:2], rv[:, 2:4]

    return compareX, compareY


def makeTestPattern(patchWidth, nbits):
    """
    Generate a set of test pairs using sampling geometry II
    # Inputs          Description
    # --------------------------------------------------------------------------
    # patchWidth      Size of the ledge of the squared patch. Suggest 9
    # nbits           Number of comparison tests. Suggest 256
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # compareX        shape (nbits, 2) matrix of (x,y) pairs of test pixel #1
    # compareY        shape (nbits, 2) matrix of (x,y) pairs of test pixel #2
    """
    # Compute test pattern using geometry II
    compareX, compareY = makeTestPatternII(patchWidth, nbits)

    return compareX, compareY


def remove_indescribable(im, locsDoG, compareXY):
    """
    Remove keypoints who cannot be BRIEF-described, because at least one of
        their comparison tests will fall out of bounds
    # Inputs            Description
    # --------------------------------------------------------------------------
    # im                Grayscale image with range [0,1].
    # locsDoG           N x 3 matrix where the DoG pyramid achieves a local extrema
    #                       in both scale and space, and satisfies the two thresholds.
    # compareXY         (2 * nbits) x 2 matrix of (x,y) pairs
    #
    # Outputs           Description
    # --------------------------------------------------------------------------
    # locs              A version of locsDoG reduced to describable keypoints only
    """
    # Calculate extreme displacements (testing on most extreme cases is enough)
    dXY_min, dXY_max = compareXY.min(axis=0), compareXY.max(axis=0)

    # Convert to indices AFTER applying extreme displacement
    indices_min = my_sift.keypoints2indices(locsDoG + np.append(dXY_min, 0))
    indices_max = my_sift.keypoints2indices(locsDoG + np.append(dXY_max, 0))

    # Compute within-bounds mask
    mask_min = mask_within_bounds(im, indices_min[:, 1:3], is_negative_ok=False)
    mask_max = mask_within_bounds(im, indices_max[:, 1:3], is_negative_ok=False)

    # Remove indescribable keypoints
    locs = locsDoG[mask_min & mask_max]

    return locs


def computeBriefOfPoint(GaussianPyramid, keypoint_index, compareX, compareY):
    """
    compute the BRIEF descriptor for provided keypoints
    # Inputs          Description
    # --------------------------------------------------------------------------
    # GaussianPyramid shape (len(levels), shape(im))matrix of the Gaussian
    #                 Pyramid created by Gaussian blurring with a different
    #                 sigma for each layer
    # keypoint_index  shape (3,) matrix of a keypoint as (lvl_idx,x,y)
    # compareX        shape (nbits, 2) matrix of (x,y) pairs of test pixel #1
    # compareY        shape (nbits, 2) matrix of (x,y) pairs of test pixel #2
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # keypoint_desc   A BRIEF descriptor of that single keypoint, shape (1, nbits)
    """
    # Compute X and Y *arrays* in indexing format, by adding dX, dY vectorially
    # (KP_lvl, KP_x, KP_y) + (0, dx, dy)
    X = keypoint_index + np.pad(compareX, ((0, 0), (1, 0)))
    Y = keypoint_index + np.pad(compareY, ((0, 0), (1, 0)))

    # Perform comparison vectorially
    keypoint_desc = (GaussianPyramid[tuple(X.T)] < GaussianPyramid[tuple(Y.T)])

    return keypoint_desc


def computeBrief(im, GaussianPyramid, locsDoG, k, levels, compareX, compareY):
    """
    compute the BRIEF descriptor for provided keypoints
    # Inputs          Description
    # --------------------------------------------------------------------------
    # im              Grayscale image with range [0,1].
    # GaussianPyramid shape (len(levels), shape(im))matrix of the Gaussian
    #                 Pyramid created by Gaussian blurring with a different
    #                 sigma for each layer
    # locsDoG         shape (?, 3) matrix of keypoints found by our DoGdetector
    # levels          Levels of Gaussian Blurring corresponding the pyramid
    # compareX        shape (nbits, 2) matrix of (x,y) pairs of test pixel #1
    # compareY        shape (nbits, 2) matrix of (x,y) pairs of test pixel #2
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # locs            A version of locsDoG reduced to BRIEF-describable
    #                 keypoints only, shape (m, 3)
    # desc            locs corresponding BRIEF-descriptions, shape (m, nbits)
    """
    # Remove keypoints which cannot be BRIEF described, because at least one of
    #   their comparison tests will fall out of bounds
    locs = remove_indescribable(im, locsDoG, np.vstack([compareX, compareY]))

    # Convert locsDoG (x,y,lvl) format to tensor indexing format (lvl-idx,y,x)
    indices = my_sift.keypoints2indices(locs, levels)

    # Prepare complete test for single point anonymous function
    test = lambda index: computeBriefOfPoint(GaussianPyramid, index,
                                             compareX, compareY)

    # Compare between the Gaussian Pyramid's intensities at each X vs Y point
    desc = np.array(list(map(test, indices)))

    return locs, desc


def briefLite(im):
    """
    compute the BRIEF descriptor for provided keypoints
    # Inputs          Description
    # --------------------------------------------------------------------------
    # im              Grayscale image with range [0,1].
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # locs            A version of locsDoG reduced to BRIEF-describable
    #                 keypoints only, shape (m, 3)
    # desc            locs corresponding BRIEF-descriptions, shape (m, nbits)
    """
    # 0) Safely, if necessary, convert image to grayscale, float32 within [0,1]
    im_gs = my_sift.bgr2gray(skimage.img_as_float32(im))

    # 1) Load test pattern and other parameters from .mat file
    dloadmat = sio.loadmat(MAT_FNAME)
    sigma0 = np.squeeze(dloadmat["sigma0"])
    k = np.squeeze(dloadmat["k"])
    levels = np.squeeze(dloadmat["levels"])
    th_contrast = np.squeeze(dloadmat["th_contrast"])
    th_r = np.squeeze(dloadmat["th_r"])
    compareX, compareY = np.squeeze(dloadmat["compareX"]), np.squeeze(dloadmat["compareY"])

    # 2) Detect SIFT keypoints locations in color image
    locsDoG, GaussianPyramid = \
        my_sift.DoGdetector(im, sigma0, k, levels, th_contrast, th_r)

    # 3) Compute BRIEF descriptor to all keypoints (with valid descriptors only!)
    locs, desc = \
        computeBrief(im_gs, GaussianPyramid, locsDoG, k, levels, compareX, compareY)

    return locs, desc


def briefMatch(desc1, desc2, ratio):
    """
    Performs the descriptor matching
        inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the
                  number of keypoints in image 1 and 2. n is the number of
                  bits in the brief
                  ratio - A match-certainty threshold factor, in [0,1]
        outputs : matches - p x 2 matrix. where the first column are indices
                  into desc1 and the second column are indices into desc2
    """
    # Try computing cdist in dtype=np.bool_ first for better efficiency
    try:
        D = cdist(desc1, desc2, metric='hamming')
    except:
        D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:, 0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1, ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    """
    Display matched points
    # Inputs          Description
    # --------------------------------------------------------------------------
    # im1             First grayscale image with range [0,1].
    # im2             Second grayscale image with range [0,1].
    # matches         Array of matches returned by briefMatch
    # locs1           Locations of keypoints returned by briefLite(im1)
    # locs2           Locations of keypoints returned by briefLite(im2)
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # fig             A figure handle to allow automated saving outside this
    #                 function
    """
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i, 0], 0:2]
        pt2 = locs2[matches[i, 1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x, y, 'r')
        plt.plot(x, y, 'g.')
    plt.show()

    return fig


def plotMatchesJet(im1, im2, matches, locs1, locs2):
    """
    Display matched points
    # Inputs          Description
    # --------------------------------------------------------------------------
    # im1             First grayscale image with range [0,1].
    # im2             Second grayscale image with range [0,1].
    # matches         Array of matches returned by briefMatch
    # locs1           Locations of keypoints returned by briefLite(im1)
    # locs2           Locations of keypoints returned by briefLite(im2)
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # fig             A figure handle to allow automated saving outside this
    #                 function
    """
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    cmap = cm.jet(np.linspace(0, 1, matches.shape[0]))
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i, 0], 0:2]
        pt2 = locs2[matches[i, 1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x, y, color=cmap[i], linewidth=1)
        plt.plot(x, y, 'g.')
    plt.show()

    return fig


def displayTestPattern(patchWidth, compareX, compareY):
    """Draw arrows on a patch grid from compareX to compareY"""
    # Create axes
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_aspect("equal")
    # Inverse y-axss, to match image axes
    ax.set_xlim(-0.5 * patchWidth, 0.5 * patchWidth)
    ax.set_ylim(0.5 * patchWidth, -0.5 * patchWidth)
    ax.grid()
    ax.set_xlabel(r"$\Delta$x (Column)")
    ax.set_ylabel(r"$\Delta$y (Row)")
    ax.set_title(r"test pattern")

    # Add arrows
    cmap = cm.jet(np.linspace(0, 1, len(compareX)))
    for i, (X, Y) in enumerate(zip(compareX, compareY)):
        ax.arrow(*X, *(Y-X), length_includes_head=True,
                 color=cmap[i], head_width=0.15)

    return fig


def rotationMatch(im_fname=r"chickenbroth_01.jpg"):
    """
    Display matched points for a rotated image
    # Input         Description
    # --------------------------------------------------------------------------
    # im_fname      Name of the image
    """
    # Load image
    im1 = cv2.imread(str(DATA_FOLDER / im_fname))
    # Fully perform SIFT detection and BRIEF description
    locs1, desc1 = briefLite(im1)
    # Setting up rotation
    im_r, im_c, _ = im1.shape
    print(f"Rotation match fo image {im_fname}:")
    degs = np.arange(10 , 90 , 10)
    for deg in degs:
        # Rotating the image
        rot_mat = cv2.getRotationMatrix2D((im_c/2,im_r/2),deg,1)
        im2 = cv2.warpAffine(im1,rot_mat,(im_c,im_r))
        # Fully perform SIFT detection and BRIEF description
        locs2 , desc2 = briefLite(im2)
        # Compute matches using the provided briefMatch function
        matches = briefMatch(desc1, desc2, ratio=1)

        # Display matched points using the provided plotMatches function
        out = f"Matches ({im_fname} at {deg} degrees), ratio={1}"
        fig = plotMatchesJet(im1, im2, matches, locs1, locs2)
        fig.axes[0].set_title(out)
        fig.savefig(OUTPUT_FOLDER / f"{out}_{deg}.png", bbox_inches="tight", pad_inches=0)
        plt.close()



    return None


def testMatch(im1_fname=r"chickenbroth_01.jpg", im2_fname=r"chickenbroth_02.jpg",
              match_ratio=MATCH_RATIO):
    """
    Display matched points
    # Inputs          Description
    # --------------------------------------------------------------------------
    # im1_name        Name of first image
    # im2_name        Name of second omage     Second grayscale image with range [0,1].
    # match_ratio     Significance factor threshold of ratio between distances of
    #                 accepted best match the next best match. Value below 1.
    #                 The closer it is to zero, the more strict its thresholding.
    #
    # Outputs         Description
    # --------------------------------------------------------------------------
    # fig             A figure handle to allow automated saving outside this
    #                 function
    """
    # Load first image
    im1 = cv2.imread(str(DATA_FOLDER / im1_fname))
    # Fully perform SIFT detection and BRIEF description
    locs1, desc1 = briefLite(im1)

    # Load second image
    im2 = cv2.imread(str(DATA_FOLDER / im2_fname))
    # Fully perform SIFT detection and BRIEF description
    locs2, desc2 = briefLite(im2)

    # Compute matches using the provided briefMatch function
    matches = briefMatch(desc1, desc2, ratio=match_ratio)

    # Display matched points using the provided plotMatches function
    out = f"Matches ({im1_fname} , {im2_fname}), ratio={match_ratio}"
    fig = plotMatchesJet(im1, im2, matches, locs1, locs2)
    fig.axes[0].set_title(out)
    fig.savefig(OUTPUT_FOLDER / f"{out}.png", bbox_inches="tight", pad_inches=0)

    return fig


###############################################################################
# -----------------------------------------------------------------------------
# When runs as a script
# -----------------------------------------------------------------------------
###############################################################################
if __name__ == "__main__":
    # A non namespace wasting stoage for Run As Script variables
    ras = RunAsScriptStore()

    # # Store general parameters in ras
    ras.sigma0 = SIGMA0
    ras.k = K
    ras.levels = LEVELS
    ras.th_contrast = TH_CONTRAST
    ras.th_r = TH_R
    ras.patchWidth = PATCH_WIDTH
    ras.nbits = NBITS

    # 2.1) If testPattern.mat does not exist or if compareX, compareY are not
    #   defined in it, generate test patterns and save to testPattern.mat
    ras.compareX, ras.compareY = None, None
    try:
        # If file not found, makeTestPattern is needed
        ras.loadmat(MAT_FNAME, variable_names=VAR_NAMES)
        # If compareX or compareY remained None, makeTestPattern is needed
        if ras.compareX is None or ras.compareY is None:
            raise AttributeError()
    except (IOError, AttributeError):
        # Generate test pattern
        ras.compareX, ras.compareY = makeTestPattern(ras.patchWidth, ras.nbits)
        # Save into testPattern.mat
        ras.savemat(MAT_FNAME)
    # # Display test pattern
    # # 2.1) Later on *always* load test pattern .mat file, and display
    ras.fig = displayTestPattern(ras.patchWidth, ras.compareX, ras.compareY)
    ras.fig.savefig(OUTPUT_FOLDER / "testPattern.png",
                    bbox_inches="tight", pad_inches=0.1)

    # 2.2) See computeBrief function

    # 2.3) Compute DoG feature points and BRIEF descriptions end-to-end
    # Load model_chickenbroth.jpg image
    ras.im = cv2.imread(str(DATA_FOLDER / r"model_chickenbroth.jpg"))
    # Convert image to grayscale and float32 within [0,1]
    ras.im_gs = my_sift.bgr2gray(skimage.img_as_float32(ras.im))
    # Fully perform a SIFT detection and BRIEF description
    ras.locs, ras.desc = briefLite(ras.im)

    # 2.4) Test the BRIEF descriptor on chickenbroth*.jpg with ratio=0.8,
    #   together with other images with other ratio values, and visualize result
    # chickenbroth, ratio=0.8
    ras.fig = testMatch()
    rotationMatch()
    plt.close(ras.fig)
    plt.clf()

    # Other tests
    ras.tests_list = \
        {(r"incline_L.png", r"incline_R.png", 0.2),
         (r"incline_L.png", r"incline_R.png", 0.4),
         (r"incline_L.png", r"incline_R.png", 0.8),
         # (r"incline_L.png", r"incline_R.png", 1.6),
         (r"pf_scan_scaled.jpg", r"pf_desk.jpg", 0.8),
         (r"pf_scan_scaled.jpg", r"pf_floor.jpg", 0.8),
         (r"pf_scan_scaled.jpg", r"pf_floor_rot.jpg", 0.8),
         (r"pf_scan_scaled.jpg", r"pf_pile.jpg", 0.8),
         (r"pf_scan_scaled.jpg", r"pf_stand.jpg", 0.8),
         # (r"pf_scan_scaled.jpg", r"pf_desk.jpg", 0.2),
         # (r"pf_scan_scaled.jpg", r"pf_floor.jpg", 0.2),
         # (r"pf_scan_scaled.jpg", r"pf_floor_rot.jpg", 0.2),
         # (r"pf_scan_scaled.jpg", r"pf_pile.jpg", 0.2),
         # (r"pf_scan_scaled.jpg", r"pf_stand.jpg", 0.2),
         # (r"Gonen20_L 400 x 300.jpg", r"Gonen20_R 400 x 300.jpg", 0.8),
         (r"Panoramic Desert 400 x 300.jpg", r"Panoramic Desert 1000 x 750.jpg", 0.4),
         }
    for ras.i, ras.test in enumerate(ras.tests_list):
        # Print progress
        print(f"Next: {ras.i + 1}/{len(ras.tests_list)}")
        # Run test and time
        start = time.time()
        ras.fig = testMatch(*ras.test)
        if ras.i in [1,6]:
            rotationMatch(ras.test[0])
        plt.close(ras.fig)
        plt.clf()
        print(time.time() - start)
    print("All Done!")

    # bonus task
