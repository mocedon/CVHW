import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh

#Add imports if needed:
from my_homography import imread
import os
import glob
from frame_video_convert import video_to_image_seq as vid2ims
from frame_video_convert import image_seq_to_video as ims2vid
#end imports

#Add functions here:
def imwrite(im, path, note="new"):
    # Write the name with suffix
    p, ext = os.path.splitext(path)
    dir, fn = os.path.split(p)
    out_dir = '../output'
    path_new = os.path.join(out_dir, fn + note + ext)
    cv2.imwrite(path_new, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

#Functions end

# HW functions:
def create_ref(im_path):
    """ create_ref(im_path) -> im"""
    im = imread(im_path)
    H, W = im.shape[:2]
    plt.imshow(im)
    print("Pick points by the following order:\n"
          "Top Left\n"
          "Top Right\n"
          "Bottom Left\n"
          "Bottom Right\n"
          "Top Middle\n"
          "Right Middle")
    # p1 are know in advance
    p1 = [[0  , 0  ],
          [W-1 , 0  ],
          [0  , H-1],
          [W-1, H-1],
          [W/2, 0  ],
          [W-1, H/2]]
    p2 = plt.ginput(6, timeout=0)

    p1 = np.vstack(p1).T
    p2 = np.vstack(p2).T
    H = mh.computeH(p1, p2) # calc Mat H

    sz = im.shape

    ref_image = mh.warpH(im, H, sz)

    return ref_image


def im2im(base, ref, item):

    p1, p2 = mh.getPoints_SIFT(base, ref) # find ref in base
    H = mh.ransacH(p1, p2)
    sz = (ref.shape[1], ref.shape[0])
    item_rs = cv2.resize(item, sz, interpolation=cv2.INTER_LINEAR)
    item_w = mh.warpH(item_rs, H, base.shape)
    im = base
    mask = np.any(item_w[:, :] != [0, 0, 0], axis=2) # object mask
    im[mask] = item_w[mask] # place the image

    return im


def vid2vid(main, ref, side):
    out = []
    length = min(len(main), len(side)) # Take the shorter
    for i in range(length):
        out.append(im2im(main[i], ref, side[i]))  # im2im on each pair of frames
    return out


if __name__ == '__main__':
    print('my_ar')

    # For debugging
    sections2run = ['Q3.1 v',
                    'Q3.2 v',
                    'Q3.3 v',
                    'Q3.4 ']

    if 'Q3.1 v' in sections2run:
        print("Start Q3.1")
        paths = [#'./my_data/book1.jpg',
                 './my_data/book2.jpg',
                 './my_data/book3.jpg',
                 './my_data/book4.jpg']
        for path in paths:
            im = create_ref(path)

            imwrite(im, path, "_ref")


    if 'Q3.2 v' in sections2run:
        print("Start Q3.2")

        ref = imread('../output/book4_ref.jpg')
        img_pairs = [['./my_data/bet1.jpg', '../output/book3_ref.jpg'],
                     ['./my_data/bet2.jpg', '../output/book2_ref.jpg'],
                     ['./my_data/bet3.jpg', '../output/book2_ref.jpg'],
                     ['./my_data/bet4.jpg', '../output/book3_ref.jpg'],
                     ['./my_data/bet5.jpg', '../output/book3_ref.jpg']]
        i = 0
        for base_p, inst_p in img_pairs:
            item = imread(inst_p)
            base = imread(base_p)
            im = im2im(base, ref, item)

            cv2.imwrite(f'../output/im2im{i}.jpg', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    if 'Q3.3 v' in sections2run:
        print("Start Q3.2")
        vid_main_path = './my_data/vid_main.mp4'
        ims_main_path = './my_data/vid_main_dir'
        vid_side_path = './my_data/vid_side.mp4'
        ims_side_path = './my_data/vid_side_dir'
        vid_outp_path = '../output/vid_outp_dir'
        vid_finl_path = '../output/vid2vid.mp4'

        vid2ims(vid_main_path, ims_main_path)
        vid2ims(vid_side_path, ims_side_path)

        ref = imread('../output/book4_ref.jpg')

        ims_main = []
        for f in glob.glob(os.path.join(ims_main_path, '*.jpg')):
            ims_main.append(imread(f))

        ims_side = []
        for f in glob.glob(os.path.join(ims_side_path, '*.jpg')):
            ims_side.append(imread(f))

        ims_out = vid2vid(ims_main, ref, ims_side)

        if not os.path.isdir(vid_outp_path):
            os.mkdir(vid_outp_path)
        for i, im in enumerate(ims_out):
            p = os.path.join(vid_outp_path, "im" + '{:04d}'.format(i) +".jpg")
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, im)

        ims2vid(vid_outp_path, vid_finl_path)

    if 'Q3.4 v' in sections2run:
        print("Start Q3.4")

    print("done")
