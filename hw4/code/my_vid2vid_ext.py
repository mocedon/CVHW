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

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    im = imread('./my_data/p1.jpg')

    mask = dlSegment(im, model)
    plt.imshow(mask)
    plt.show()