import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh
from my_homography import imread
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from my_ar import vid2vid
from frame_video_convert import video_to_image_seq as vid2ims
from frame_video_convert import image_seq_to_video as ims2vid


def dlSegment(img, model):
    """dlSegment(img, model) => masked image"""
    """Deep Learning segmentation algorithm"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
    # perform pre-processing
    input_tensor = preprocess(img) # Img normalization
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of size 1 as expected by the model

    # send to device
    input_batch = input_batch.to(device)
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    pred = np.where(output.argmax(0) == 15, 1, 0).astype('uint8') # Pick the person
    plt.imshow(pred)
    plt.show()
    return img * pred[:, :, np.newaxis]


if __name__ == '__main__':
    print('my_vid2vid_ext')

    # model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    # model.eval()
    #
    # im = imread('./my_data/p1.jpg')
    #
    # mask = dlSegment(im, model)
    # plt.imshow(mask)
    # plt.show()
    vml = './my_data/pnl.mp4'
    iml = './my_data/pnl'

    vsl = './my_data/pnr.mp4'
    isl = './my_data/pnr'

    vmr = './my_data/pnr.mp4'
    imr = './my_data/pnr'

    vsr = './my_data/pnl.mp4'
    isr = './my_data/pnl'

    iol = '../output/iol_dir'
    ior = '../output/ior_dir'
    ioc = '../output/ioc_dir'
    vid = '../output/vid.mp4'

    vid2ims(vml, iml)
    vid2ims(vsl, isl)
    vid2ims(vmr, imr)
    vid2ims(vsr, isr)

    ref = imread('../output/book4_ref.jpg')

    # Left video creation
    ims_main = []
    for f in glob.glob(os.path.join(iml, '*.jpg')):
        ims_main.append(imread(f))

    ims_side = []
    for f in glob.glob(os.path.join(isl, '*.jpg')):
        ims_side.append(imread(f))

    img_out_l = vid2vid(ims_main[:2], ref, ims_side[:2])

    if not os.path.isdir(iol):
        os.mkdir(iol)
    for i, im in enumerate(img_out_l):
        p = os.path.join(iol, "im" + '{:04d}'.format(i) + ".jpg")
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, im)

    # Right video creation
    ims_main = []
    for f in glob.glob(os.path.join(imr, '*.jpg')):
        ims_main.append(imread(f))

    ims_side = []
    for f in glob.glob(os.path.join(isr, '*.jpg')):
        ims_side.append(imread(f))

    img_out_r = vid2vid(ims_main[:2], ref, ims_side[:2])

    if not os.path.isdir(ior):
        os.mkdir(ior)
    for i, im in enumerate(img_out_r):
        p = os.path.join(ior, "im" + '{:04d}'.format(i) + ".jpg")
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, im)

    # Stitch left video with right video
    width, height = img_out_l[0].shape[:2]

    scale = np.array([1, 2, 1])
    sz = np.array([width, height, 3] * scale)

    for i in range(min(len(img_out_l), len(img_out_r))):
        l = img_out_l[i]
        r = img_out_r[i]
        pan = np.zeros(sz)
        pan[:width, :height] = l[:width, :height]
        p1, p2 = mh.getPoints_SIFT(l, r)
        H = mh.ransacH(p1, p2)
        rw = mh.warpH(r, H, sz)
        pan = mh.imageStitching(pan, rw)
        p = os.path.join(ioc, "im" + '{:04d}'.format(i) + ".jpg")
        im = cv2.cvtColor(pan, cv2.COLOR_RGB2BGR)
        if not os.path.isdir(ioc):
            os.mkdir(ioc)
        cv2.imwrite(p, im)

    ims2vid(ioc, vid)













