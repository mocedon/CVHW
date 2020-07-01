import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh

#Add imports if needed:
from my_homography import imread
import os
#end imports

#Add functions here:
def imwrite(im, path, note="new"):
    p, ext = os.path.splitext(path)
    dir, fn = os.path.split(p)
    out_dir = '../output'
    path_new = os.path.join(out_dir, fn + note + ext)
    cv2.imwrite(path_new, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

#Functions end

# HW functions:
def create_ref(im_path):
    im = imread(im_path)
    blnk = np.zeros_like(im)
    p1, p2 = mh.getPoints(blnk, im, 6)
    print(f'p1= {p1}')
    print(f'p2= {p2}')
    H = mh.computeH(p1, p2)
    print(f'H= {H}')

    ref_image = mh.warpH(im, H, im.shape)

    return ref_image


def im2im(base, item):

    p1, p2 = mh.getPoints(base, item, 6)
    H = mh.computeH(p1, p2)
    item_w = mh.warpH(item, H, base.shape)
    im = base
    mask = np.any(item_w[:, :] != [0, 0, 0], axis=2)
    im[mask] = item_w[mask]

    return im


if __name__ == '__main__':
    print('my_ar')

    sections2run = ['Q3.1 ',
                    'Q3.2 v',
                    'Q3.3 ',
                    'Q3.4 ']
    if 'Q3.1 v' in sections2run:
        paths = [#'./my_data/sapiens1.jpg',
                 './my_data/sapiens3.jpg']
        refs = []
        for path in paths:
            im = create_ref(path)

            imwrite(im, path, "_ref")
            refs.append(im)
        plt.imshow(np.hstack(refs))
        plt.show()

    if 'Q3.2 v' in sections2run:
        img_pairs = [#['./my_data/set1.jpg', '../output/sapiens1_ref.jpg'],
                     #['./my_data/set2.jpg', '../output/sapiens3_ref.jpg'],
                     ['./my_data/set3.jpg', '../output/sapiens1_ref.jpg']]

        for base_p, inst_p in img_pairs:
            item = imread(inst_p)
            base = imread(base_p)
            im = im2im(base, item)
            imwrite(im, base_p, "_ar")



    print("done")
