import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt


# #Add imports if needed:
from scipy import interpolate as interp
import code.PythonSIFT.pysift as pysift
# #end imports
#
# #Add extra functions here:
def imread(path):
    im = cv2.imread(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# #Extra functions end

# HW functions:
def getPoints(im1, im2, N=6):
    # TODO: create a better way
    """
    getPoints(im1, im2, N) -> p1 , p2
    Manually pick of corresponding points on 2 images
    """
    plt.imshow(np.hstack([im1, im2]))
    plt.xticks([])
    plt.yticks([])
    print("Match points on both images")
    pts = plt.ginput(2 * N, timeout=0)
    plt.show()

    p1 = []
    p2 = []

    width = im1.shape[1]
    for i, pt in enumerate(pts, 1):
        if i % 2:
            p1.append(np.array(pt))
        else:
            p2.append(np.array([pt[0] - width, pt[1]]))
    p1 = np.vstack(p1).T
    p2 = np.vstack(p2).T
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
    imW = im1.shape[0]
    imH = im1.shape[1]
    for i in range(imH * imW):
        u = int(i % imW)
        v = int(np.floor(i / imW))
        if not np.all(im1[u,v,:] == [0, 0, 0]):
            src = np.array([u, v, 1])
            dst = H @ src.T
            x = int(dst[0] / dst[2])
            y = int(dst[1] / dst[2])
            if 0 <= x < out_size[0] and 0 <= y < out_size[1]:
                warp[x, y, :] = im1[u, v, :]

    # warp_lab = cv2.cvtColor(warp, cv2.COLOR_RGB2LAB)
    # lab = [warp_lab[:, :, 0], warp_lab[:, :, 1], warp_lab[:, :, 2]]
    #
    # x = np.arange(out_size[1])
    # y = np.arange(out_size[0])
    # xx, yy = np.meshgrid(x, y)
    #
    # warp_im1 = np.zeros_like(warp)
    # for i, ch in enumerate(lab):
    #     mask = ch > 0
    #     xx_used = xx[mask].ravel()
    #     yy_used = yy[mask].ravel()
    #     ch_used = ch[mask].ravel()
    #     intrp = interp.interp2d(xx_used, yy_used, ch_used, kind='linear')
    #     warp_im1[:, :, i] = intrp(x, y)
    # warp_im1 = cv2.cvtColor(warp_im1, cv2.COLOR_LAB2RGB)

    return warp


def imageStitching(img1, wrap_img2):
    #TODO: merge better
    panoImg = np.zeros_like(wrap_img2)
    im1H, im1W = img1.shape[:2]
    im2H, im2W = wrap_img2.shape[:2]
    panoImg[:im1H, :im1W, :] = img1
    mask = np.all(panoImg[:, :] == [0, 0, 0], axis=2)
    panoImg[mask] = wrap_img2[mask]
#     im1M = np.full((im2H, im2W), False)
#     im1M[:im1H, :im1W] = np.any(img1 > [0, 0, 0], axis=2)
# #    im1M = np.transpose(np.stack([im1M, im1M, im1M]), [1, 2, 0])
#     im2M = np.any(wrap_img2 > [0, 0, 0] , axis=2)
# #    im2M = np.transpose(np.stack([im2M, im2M, im2M]), [1, 2, 0])
#     #im2M = np.bitwise_xor(im1M, im2M)
#     panoImg[im1M[:, :, np.newaxis]] = img1[im1M[:, :, np.newaxis]]
#     panoImg[im2M[:, :, np.newaxis]] = wrap_img2[im2M[:, :, np.newaxis]]
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
    SFT = cv2.ORB_create()
    kp1, ds1 = SFT.detectAndCompute(im1, None)
    kp2, ds2 = SFT.detectAndCompute(im2, None)

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
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    working_on = 4
    if working_on < 2:
        if working_on == 0:
            p1,p2 = getPoints(im1, im2, 6)
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
        H = computeH(p1[:12].T, p2[:12].T)
        imW = im1.shape[0]
        imH = im1.shape[1]
        im = warpH(im2, H, (2 * imW, 2 * imH))

        pan = imageStitching(im1, im)
        plt.imshow(pan)

    if working_on == 3:
        b1 = imread('data/beach3.jpg')
        b2 = imread('data/beach2.jpg')
        b3 = imread('data/beach1.jpg')



        bW = b1.shape[0]
        bH = b1.shape[1]


        st = [b1, b2, b3]
        scale = np.array([3, 2, 1])
        sz = np.array([bW, bH,3] * scale)

        H = np.array([[1, 0, 0   ],
                      [0, 1, bH/2],
                      [0, 0, 1   ]])
        pan = np.zeros(sz)
        base = b1

        for img in st:
            p1, p2 = getPoints_SIFT(base, img)
            H = H @ ransacH(p1[:120], p2[:120])

            img_w = warpH(img, H, sz)
            plt.imshow(img_w)
            plt.show()
            pan = imageStitching(pan, img_w)
            base = img

        plt.imshow(pan)



    if working_on == 4:
        s1 = imread('data/sintra5.JPG')
        s2 = imread('data/sintra4.JPG')
        s3 = imread('data/sintra3.JPG')
        s4 = imread('data/sintra2.JPG')
        s5 = imread('data/sintra1.JPG')

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