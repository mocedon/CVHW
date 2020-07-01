import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh

#Add imports if needed:
import mh.imread as imread
#end imports

#Add functions here:
"""
   Your code here
"""
#Functions end

# HW functions:
def create_ref(im_path):
    im = imread(im_path)
    bl = np.zeros_like(im)
    p1, p2 = mh.GetPoints(im, bl, 4)
    H = mh.computeH(p1, p2)

    ref_image = mh.warpH(im, H, im.shape)

    return ref_image


if __name__ == '__main__':
    print('my_ar')
    paths = ['./my_data/sapiens1.jpg',
             './my_data/sapiens2.jpg',
             './my_data/sapiens3.jpg']

    refs = []
    for path in paths:
        im = create_ref(path)
        refs.append(im)
        plt.imshow(im)
        plt.show()

    print("done")
