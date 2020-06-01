import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def segModel():
    """Deep learning segmentation model"""
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained= True)
    model.eval()
    return model


def classModel():
    """Deep learning classification model"""
    model = torchvision.models.alexnet(pretrained= True)
    model.eval()
    return model


def showImages(lst, rows=1):
    """showImages(lst)"""
    """takes a list of images and shows them all"""
    if len(lst) == 1:
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(lst[0])
    else:
        cols = int(np.ceil(len(lst) / rows))
        fig, axes = plt.subplots(rows, cols)
        axes = axes.ravel()
        for idx, img in enumerate(lst):
            axes[idx].imshow(img)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
    plt.show()


def getImages(path):
    """getImages(path) => list of images"""
    """Takes a path and extracts all images"""
    if not os.path.exists(path):
        return []
    images = []
    files = os.listdir(path)
    for file in files:
        if ".jpg" in file or ".png" in file:
            img = cv2.imread(os.path.join(path, file))
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    showImages(images)
    return images


def classicSegment(img):
    """ClassicSegment(img) => masked image"""
    """Take an image and segments it with classic algorithm"""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, mask.shape[0] - 10, mask.shape[1] - 10)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return img * mask2[:, :, np.newaxis]


def dlSegment(img, model):
    """dlSegment(img, model) => masked image"""
    """Deep Learning segmentation algorithm"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
    # perform pre-processing
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of size 1 as expected by the model

    # send to device
    input_batch = input_batch.to(device)
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    pred = np.where(output.argmax(0) > 0, 1, 0).astype('uint8')
    return img * pred[:, :, np.newaxis]


def dlClassify(img,model):
    """dlClassify(img, model) => masked image"""
    """Deep Learning classification algorithm"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
    # perform pre-processing
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of size 1 as expected by the model

    # send to device
    input_batch = input_batch.to(device)
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        output = model(input_batch)

    cont = open("./data/imagenet1000_clsidx_to_labels.txt" , 'r').read()
    labels = eval(cont)
    return labels[int(output.argmax())]


def changeBg(fg, bg):
    fgSz = np.array(fg.shape[:2])
    bgSz = np.array(bg.shape[:2])
    if (fgSz >= bgSz).all():
        pilImg = transforms.ToPILImage()
        fg = pilImg(fg)
        scale = np.min(np.divide(bgSz, fgSz))
        fgSz = np.floor(scale * fgSz).astype(int)
        resize = transforms.Resize(fgSz)
        fg = resize(fg)
    prep = transforms.ToTensor()
#    fg = prep(fg)
#    bg = prep(bg)
    bg_ = bg[bgSz[0]-fgSz[0]:, :fgSz[1], :]
    bg_[fg > 0] = fg[fg > 0]
    bg[bgSz[0]-fgSz[0]:, :fgSz[1], :] = bg_
#    bg = np.array(np.transpose(bg, (1, 2, 0)))
#    bg = np.floor(bg * 255.99).astype(np.uint8)
#    bg[bg == 256] = 255
    return bg


def main():
    # Get and show the images
    # frg = getImages("./data/frogs")
    # hrs = getImages("./data/horses")
    # imgs = frg + hrs
    #
    # # Classic segmentation
    # clSeg = []
    # for img in imgs:
    #     clSeg.append(classicSegment(img))
    #
    #
    #
    # # DL segmentation
    model = segModel()
    # dlSeg = []
    # for img in imgs:
    #     dlSeg.append(dlSegment(img, model))
    #
    # showImages(imgs + clSeg + dlSeg, 3)
    # # Get new images
    # myPics = getImages("./my_data/Items")
    # clSeg = []
    # dlSeg = []
    #
    # # Segment image
    # for pic in myPics:
    #     clSeg.append(classicSegment(pic))
    #     dlSeg.append(dlSegment(pic, model))
    # showImages(myPics + clSeg + dlSeg, 3)

    # pre or post-processing


    # Fish out of the water
    goat = getImages("./my_data/goat")
    classMod = classModel()
    pred = dlClassify(goat[1], classMod)
    print(pred)
    kang = dlSegment(goat[1], model)
    showImages([kang])

    kang = changeBg(kang, goat[2])
    showImages([kang])
    newPred = dlClassify(kang, classMod)
    cv2.imwrite("../output/Kangaroo_in_autria.jpg", cv2.cvtColor(kang, cv2.COLOR_RGB2BGR))
    print(newPred)
    print("All done!")


if __name__ == "__main__":

    main()