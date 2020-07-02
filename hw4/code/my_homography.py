import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt


# #Add imports if needed:
from scipy import interpolate as interp
import PythonSIFT.pysift as pysift
# #end imports
#789
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
    # Plot images
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Pick a key-point HERE first")
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Pick a corresponding key-point HERE next")
    print("Match points on both images, left-right-left-right...")

    # Choose points
    pts = plt.ginput(2 * N, timeout=0)

    # Separate between images and stack as matrix
    pts1 = np.row_stack(pts[::2]).T
    pts2 = np.row_stack(pts[1::2]).T

    return p1, p2


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
    # TODO: get interpolation working
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

    return warp


def imageStitching(img1, wrap_img2):
    #TODO: merge better
    panoImg = np.zeros_like(wrap_img2)
    mask1 = np.any(img1[:, :] > [0, 0, 0], axis=2)
    mask2 = np.any(wrap_img2[:, :] > [0, 0, 0], axis=2)
    mask0 = np.bitwise_and(mask1, mask2) # Mask overlap
    mask1 = np.bitwise_xor(mask1,mask0)
    mask2 = np.bitwise_xor(mask2,mask0)
    panoImg[mask1] = img1[mask1]
    panoImg[mask2] = wrap_img2[mask2]
    panoImg[mask0] = img1[mask0]
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
    plt.imshow(img3), plt.show()

    return p1, p2


if __name__ == '__main__':
    print('my_homography')
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
        imW =im1.shape[0]
        imH = im1.shape[1]
        im = warpH(im2, H, (2*imW, 2*imH))
        # fig, (ax1, ax2) = plt.subplots(1,2)
        # ax1.imshow(im1)
        # ax2.imshow(im)

        pan = imageStitching(im1 , im)
        plt.imshow(pan)

    if working_on == 2:
        p1, p2 = getPoints_SIFT(im1, im2)
        I = np.identity(3)
        H = computeH(p1[:12].T, p2[:12].T)
        sz = np.array(im1.shape[:2]) *2

        # H = np.array([[ 9.861e-3, -1.018e-3, -3.776e-2],
        #               [-9.200e-5,  7.204e-3,  9.991e-1],
        #               [ 1.011e-6, -1.672e-5,  1.115e-2]])
        print(f'H {H}')

        im1_p = warpH(im1, I, sz)
        im2_p = warpH(im2, H, sz)

        plt.imshow(np.hstack([im1_p, im2_p]))

        pan = imageStitching(im1_p, im2_p)
        plt.imshow(pan)

    if working_on == 3:
        b1 = imread('data/beach5.jpg', ds=5)
        b2 = imread('data/beach4.jpg', ds=5)
        b3 = imread('data/beach3.jpg', ds=5)
        b4 = imread('data/beach2.jpg', ds=5)
        b5 = imread('data/beach1.jpg', ds=5)

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
        s1 = imread('data/sintra5.JPG', ds=6)
        s2 = imread('data/sintra4.JPG', ds=6)
        s3 = imread('data/sintra3.JPG', ds=6)
        s4 = imread('data/sintra2.JPG', ds=6)
        s5 = imread('data/sintra1.JPG', ds=6)

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
                p1, p2 = getPoints_SIFT(base, img)
                H = H @ ransacH(p1[:120], p2[:120])

                img_w = warpH(img, H, sz)
                plt.imshow(img_w)
                plt.show()
                pan = imageStitching(pan, img_w)
                base = img

        plt.imshow(pan)





    # Alwatys last
    plt.show()
    print("Done")