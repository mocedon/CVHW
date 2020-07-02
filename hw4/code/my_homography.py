import pathlib
import datetime
import numpy as np
import scipy
from matplotlib import pyplot as plt
import cv2
from scipy import interpolate as interp
import PythonSIFT.pysift as pysift

OUTPUT_FOLDER = pathlib.Path(r"").absolute().parent / "output"
DS_INCLINE = 4
DS_FLOOR = 10
DS_BEACH = 15
DS_SINTRA = 15

# #Add extra functions here:
def imread(path, ds=1):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if ds > 1:
        sz = np.floor(np.array(im.shape[:2]) / ds).astype(int)
        sz = (sz[1], sz[0])
        im = cv2.resize(im, sz, interpolation=cv2.INTER_LINEAR)
    return im
# #Extra functions end

# HW functions:
def getPoints(im1, im2, N=6):
    """
    getPoints(im1, im2, N) -> p1 , p2
    Manually pick corresponding points on 2 images
    """
    # Full screen figure
    plt.figure()
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    # Plot images
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Pick a key-point HERE first", fontsize=12)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Pick a corresponding key-point HERE next", fontsize=12)
    print("Match points on both images, left-right-left-right...")

    # Choose points
    pts = plt.ginput(2 * N, timeout=0)

    # Close figure
    plt.close()

    # Separate between images and stack as matrix
    pts1 = np.row_stack(pts[::2]).T
    pts2 = np.row_stack(pts[1::2]).T

    return pts1, pts2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    A = []
    for i in range(p1.shape[1]):
        x = p1[1][i]
        y = p1[0][i]
        u = p2[1][i]
        v = p2[0][i]
        A.append(np.array([u, v, 1, 0, 0, 0, -u * x, -v * x, -x]))
        A.append(np.array([0, 0, 0, u, v, 1, -u * y, -v * y, -y]))
    A = np.vstack(A)
    (U, D, Vh) = np.linalg.svd(A, False)
    h = Vh.T[:, -1]
    H2to1 = h.reshape((3, 3))

    return H2to1


def warpH(im1, H, out_size):
    im1 = im1.copy()
    # Convert to LAB color space
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB)
    warp = np.zeros([out_size[0], out_size[1], im1.shape[2]], dtype=np.uint8)
    imW, imH = im1.shape[:2]

    ui = np.arange(imH)
    vi = np.arange(imW)

    warpW, warpH = warp.shape[:2]
    Hinv = np.linalg.inv(H)

    for ch in range(im1.shape[2]):
        intrp = interp.interp2d(ui, vi, im1[:,:,ch], kind='linear')
        for i in range(warpH*warpW):
            x = int(i % warpW)
            y = int(np.floor(i/warpW))
            dst = np.array([x, y, 1])
            src = Hinv @ dst.T
            v = src[0] / src[2]
            u = src[1] / src[2]
            if 0 <= u < imH and 0 <= v < imW:
                z = intrp(u, v)
                warp[x, y, ch] = z

    # Convert back to RGB color space, and fix "Blue background" issue
    mask = (warp == [0, 0 ,0])
    warp = cv2.cvtColor(warp, cv2.COLOR_LAB2RGB)
    warp[mask] = 0

    fig = plt.figure()
    plt.imshow(warp)
    fig.savefig((OUTPUT_FOLDER / f"warpH_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)

    return warp


def imageStitching(img1, wrap_img2):
    panoImg = np.zeros_like(wrap_img2)
    mask1 = np.any(img1[:, :] > [0, 0, 0], axis=2)
    mask2 = np.any(wrap_img2[:, :] > [0, 0, 0], axis=2)
    mask0 = np.bitwise_and(mask1, mask2) # Mask overlap
    mask1 = np.bitwise_xor(mask1,mask0)
    mask2 = np.bitwise_xor(mask2,mask0)
    panoImg[mask1] = img1[mask1]
    panoImg[mask2] = wrap_img2[mask2]
    panoImg[mask0] = img1[mask0]

    fig = plt.figure()
    plt.imshow(panoImg)
    fig.savefig((OUTPUT_FOLDER / f"warpH_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)

    return panoImg


def ransacH(p1, p2, nIter=1000, tol=10):
    bestH = []
    inline = 0
    for i in range(nIter):
        prob = np.arange(len(p1)-1, -1, -1) / np.sum(np.arange(len(p1)))
        rndIdx = np.random.choice(len(p1), 8, p=prob, replace=False)
        H = computeH(p1[rndIdx].T, p2[rndIdx].T)
        fit = 0
        for j, p in enumerate(p2):
            p = np.array([p[1], p[0], 1])
            pc = H @ p.T
            p1_clc = np.array([pc[1] / pc[2], pc[0] / pc[2]])
            dist = np.linalg.norm(p1_clc - p1[j])
            if dist < tol:
                fit += 1
        if fit > inline:
            bestH = H
            inline = fit

    #debug
    print(bestH)
    print(inline)

    return bestH


def getPoints_SIFT(im1,im2):
    # SFT = cv2.ORB_create()
    # kp1, ds1 = SFT.detectAndCompute(im1, None)
    # kp2, ds2 = SFT.detectAndCompute(im2, None)
    im1g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    kp1, ds1 = pysift.computeKeypointsAndDescriptors(im1g)
    kp2, ds2 = pysift.computeKeypointsAndDescriptors(im2g)

    BFM = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=True)

    prs = BFM.match(ds1, ds2)
    prs = sorted(prs, key=lambda x:x.distance)


    p1 = []
    p2 = []
    for m in prs:
        p1.append(kp1[m.queryIdx].pt)
        p2.append(kp2[m.trainIdx].pt)

    p1 = np.array(p1, dtype=int)
    p2 = np.array(p2, dtype=int)

    # Debug
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, prs, None, flags=2)
    fig = plt.figure()
    plt.imshow(img3)
    fig.savefig((OUTPUT_FOLDER / f"getPoints_SIFT_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)

    return p1, p2


def computeA(p1, p2):
    """Compute best fitting affine transformation between two point arrays"""
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    A = []
    b = []
    for i in range(p1.shape[1]):
        x = p1[1][i]
        y = p1[0][i]
        u = p2[1][i]
        v = p2[0][i]
        A.append(np.array([u, v, 1, 0, 0, 0]))
        A.append(np.array([0, 0, 0, u, v, 1]))
        b.append(x)
        b.append(y)

    A = np.vstack(A)
    b = np.vstack(b)
    x = np.column_stack([np.array([1, 0, 0, 0, 1, 0])])
    A2to1 = np.linalg.pinv(A) @ b
    A2to1 = np.row_stack([A2to1.reshape((2, 3)), [0, 0, 1]])

    return A2to1


def imageStitchingHomography(img1, img2, getPointsBy="ginput"):
    """Image stitching using homography transformation"""
    if getPointsBy == "SIFT":
        p1, p2 = getPoints_SIFT(img1, img2)
    else:
        p1, p2 = getPoints(img1, img2)
        p1 = p1.T
        p2 = p2.T
    I = np.identity(3)
    H = computeH(p1[:12].T, p2[:12].T)
    sz = np.array(img1.shape[:2]) * 2

    print(f'H {H}')

    img1_p = warpH(img1, I, sz)
    img2_p = warpH(img2, H, sz)

    fig = plt.figure()
    plt.imshow(np.hstack([img1_p, img2_p]))
    fig.savefig((OUTPUT_FOLDER / f"Homographically-Transformed_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)

    pan = imageStitching(img1_p, img2_p)
    plt.imshow(pan)
    fig.savefig((OUTPUT_FOLDER / f"Homographically-Stitched_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)


def imageStitchingAffine(img1, img2, getPointsBy="ginput"):
    """Image stitching using affine transformation"""
    if getPointsBy == "SIFT":
        p1, p2 = getPoints_SIFT(img1, img2)
    else:
        p1, p2 = getPoints(img1, img2)
        p1 = p1.T
        p2 = p2.T
    I = np.identity(3)
    A = computeA(p1[:12].T, p2[:12].T)
    sz = np.array(img1.shape[:2]) * 2

    print(f'A {A}')

    img1_p = warpH(img1, I, sz)
    img2_p = warpH(img2, A, sz)

    fig = plt.figure()
    plt.imshow(np.hstack([img1_p, img2_p]))
    fig.savefig((OUTPUT_FOLDER / f"Affinally-Transformed_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)

    pan = imageStitching(img1_p, img2_p)
    plt.imshow(pan)
    fig.savefig((OUTPUT_FOLDER / f"Affinally-Stitched_{datetime.datetime.now().strftime('%H%M%S')}").with_suffix(".jpg"),
                bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    print('my_homography')
    # Qyestion to be tested
    q = 22
    debug = True

    im1 = imread('data/incline_L.png', ds=DS_INCLINE)
    im2 = imread('data/incline_R.png', ds=DS_INCLINE)

    # Q2.1
    if q == 21:
        p1, p2 = getPoints(im1, im2, 6)
        print(f'p1 {p1}')
        print(f'p2 {p2}')
    else:
        p1 = np.array([[452, 610, 622, 401, 914, 414],
                       [123, 195, 489, 176, 360, 360]])
        p2 = np.array([[117, 290, 315, 58, 572, 80],
                       [152, 245, 539, 221, 398, 426]])

    # Q2.2
    if q == 22:
        im1 = imread('data/incline_L.png', ds=4)
        im2 = imread('data/incline_R.png', ds=4)


    working_on = 0

    if working_on < 2:
        if working_on == 0:
            p1, p2 = getPoints(im1, im2, 6)
            print(f'p1 {p1}')
            print(f'p2 {p2}')
        else:
            p1 = np.array([[452, 610, 622, 401, 914, 414],
                           [123, 195, 489, 176, 360, 360]])
            p2 = np.array([[117, 290, 315, 58, 572, 80],
                           [152, 245, 539, 221, 398, 426]])
        H = computeH(p1, p2)
        print(H)
        imW = im1.shape[0]
        imH = im1.shape[1]
        im = warpH(im2, H, (2*imW, 2*imH))
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(im1)
        ax2.imshow(im)
        pan = imageStitching(im1 , im)
        plt.imshow(pan)

    working_on = 2

    if working_on == 2:
        # Example of non-working affine stitching
        # print("Stitching incline-*.jpg using homography transformation")
        # imageStitchingHomography(im1, im2)
        # print("Stitching incline-*.jpg using affine transformation")
        # imageStitchingAffine(im1, im2)

        # # Debug computeA
        # # Identity
        # p1 = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        # p2 = p1.copy()
        # res = computeA(p1, p2)
        # # Mirror
        # p1 = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        # p2 = -p1.copy()
        # res = computeA(p1, p2)
        # Example of working affine stitching
        im1_ah = imread(r'my_data\FloorEqualDepth\FloorEqualDepth-L.jpg', ds=DS_FLOOR)
        im2_ah = imread(r'my_data\FloorEqualDepth\FloorEqualDepth-R.jpg', ds=DS_FLOOR)
        print("Stitching FloorEqualDepth-*.jpg using homography transformation")
        imageStitchingHomography(im1_ah, im2_ah)
        print("Stitching FloorEqualDepth-*.jpg using affine transformation")
        imageStitchingAffine(im1_ah, im2_ah)



    if working_on == 3:
        print("SIFT-demo on beach*.jpg")
        b1 = imread('data/beach5.jpg', ds=DS_BEACH)
        b2 = imread('data/beach4.jpg', ds=DS_BEACH)
        b3 = imread('data/beach3.jpg', ds=DS_BEACH)
        b4 = imread('data/beach2.jpg', ds=DS_BEACH)
        b5 = imread('data/beach1.jpg', ds=DS_BEACH)

        bW = b1.shape[0]
        bH = b1.shape[1]


        sts = [[b3, b2, b1],
              [b3, b4, b5]]

        scale = np.array([5, 2, 1])
        sz = np.array([bW, bH,3] * scale)


        pan = np.zeros(sz)
        base = b3

        for st in sts:
            H = np.array([[1, 0, bW * 2],
                          [0, 1, bH / 2],
                          [0, 0, 1]])
            base = b3

            for img in st:
                print("getPoint_SIFT")
                p1, p2 = getPoints_SIFT(base, img)
                H = H @ ransacH(p1[:120], p2[:120])

                img_w = warpH(img, H, sz)
                plt.imshow(img_w)
                plt.show()
                pan = imageStitching(pan, img_w)
                base = img


        plt.imshow(pan)
        plt.show()



    if working_on == 3:
        print("SIFT-demo on sintra*.jpg")
        s1 = imread('data/sintra5.JPG', ds=DS_SINTRA)
        s2 = imread('data/sintra4.JPG', ds=DS_SINTRA)
        s3 = imread('data/sintra3.JPG', ds=DS_SINTRA)
        s4 = imread('data/sintra2.JPG', ds=DS_SINTRA)
        s5 = imread('data/sintra1.JPG', ds=DS_SINTRA)

        sW = s1.shape[0]
        sH = s1.shape[1]

        sts = [[s3, s4, s5],
               [s3, s2, s1]]

        scale = np.array([2, 5, 1])
        sz = np.array([sW, sH, 3] * scale)
        pan = np.zeros(sz)
        plt.imshow(pan)
        plt.show()

        for st in sts:
            H = np.array([[1, 0, sW / 2],
                          [0, 1, sH * 2],
                          [0, 0, 1]])

            base = st[0]

            for img in st:
                print("getPoint_SIFT")
                p1, p2 = getPoints_SIFT(base, img)
                H = H @ ransacH(p1[:120], p2[:120])

                img_w = warpH(img, H, sz)
                plt.imshow(img_w)
                plt.show()
                pan = imageStitching(pan, img_w)
                base = img

        plt.imshow(pan)

    if working_on == 6:
        lr1 = imread('my_data/lr1.JPG')
        lr2 = imread('my_data/lr2.JPG')


        lrW = lr1.shape[0]
        lrH = lr2.shape[1]

        sts = [[lr1, lr2]]

        scale = np.array([2, 1, 1])
        sz = np.array([lrW, lrH, 3] * scale)
        pan = np.zeros(sz)


        for st in sts:
            H = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

            base = st[0]

            for img in st:
                p1, p2 = getPoints_SIFT(base, img)
                H = H @ ransacH(p1[:120], p2[:120])

                img_w = warpH(img, H, sz)
                plt.imshow(img_w)
                plt.show()
                pan = imageStitching(pan, img_w)
                base = img
        plt.imshow(pan)
        plt.show()







    # Alwatys last
    plt.show()
    print("Done")
