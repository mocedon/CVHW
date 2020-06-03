import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms


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
    if len(lst) == 1: # In case there is only 1 image to show
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(lst[0])
    else:
        cols = int(np.ceil(len(lst) / rows)) # Gets the number of columns
        fig, axes = plt.subplots(rows, cols)
        axes = axes.ravel() # avoid different case for col > 1
        for idx, img in enumerate(lst):
            axes[idx].imshow(img)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
    plt.show()


def getImages(path):
    """getImages(path) => list of images"""
    """Takes a path and extracts all images (jpg, png only)"""
    if not os.path.exists(path): # Check path
        print("Path doesn't exist")
        return []
    images = []
    files = os.listdir(path) # All files in path
    for file in files:
        if ".jpg" in file or ".png" in file: # avoid using wrong files,
            img = cv2.imread(os.path.join(path, file))
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    showImages(images)
    return images


def classicSegment(img):
    """ClassicSegment(img) => masked image"""
    """Take an image and segments it with classic algorithm"""
    # Used by the algorithm
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, mask.shape[0] - 10, mask.shape[1] - 10) # mask of most of the image, worked best for me
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') # take what's labeled fg or probably fg
    return img * mask2[:, :, np.newaxis]


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
    pred = np.where(output.argmax(0) > 0, 1, 0).astype('uint8') # Pick everywhere that isn't label 0 (bg)
    return img * pred[:, :, np.newaxis]


def dlClassify(img,model):
    """dlClassify(img, model) => masked image"""
    """Deep Learning classification algorithm"""
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
        output = model(input_batch)

    cont = open("./data/imagenet1000_clsidx_to_labels.txt" , 'r').read() # Get labels
    labels = eval(cont) # Create Dict
    return labels[int(output.argmax())] # Pick highest probability label


def changeBg(fg, bg):
    fgSz = np.array(fg.shape[:2]) # (W,H)
    bgSz = np.array(bg.shape[:2]) # (W,H)
    if (fgSz >= bgSz).all(): # Make sure fg fit in bg
        pilImg = transforms.ToPILImage()
        fg = pilImg(fg) # format for resize
        scale = np.min(np.divide(bgSz, fgSz)) # find down sample scale
        fgSz = np.floor(scale * fgSz).astype(int)
        resize = transforms.Resize(fgSz)
        fg = resize(fg)
        tsrImg = transforms.ToTensor() # format back to tensor
        tsrImg(fg)
        fg = np.transpose(np.array(fg), (0, 1 ,2)) # rearrange dims to fit imshow
    bg_ = bg[bgSz[0]-fgSz[0]:, :fgSz[1], :] # take the overlap
    bg_[fg > 0] = fg[fg > 0] # apply the new image with mask
    bg[bgSz[0]-fgSz[0]:, :fgSz[1], :] = bg_ # get overlap back to original img

    return bg


def main():
    # Get and show the images
    frg = getImages("./data/frogs") # Frogs
    hrs = getImages("./data/horses") # Horses
    imgs = frg + hrs

    # Classic segmentation
    clSeg = []
    for img in imgs:
        clSeg.append(classicSegment(img)) # Classic segmentation for all imgs

    # DL segmentation
    model = segModel()
    dlSeg = []
    for img in imgs:
        dlSeg.append(dlSegment(img, model)) # Deep Learning segmentation

    showImages(imgs + clSeg + dlSeg, 3) # Show it all in 3 rows
    # Get new images
    myPics = getImages("./my_data/Items")
    clSeg = []
    dlSeg = []

    # Segment image
    for pic in myPics:
        clSeg.append(classicSegment(pic))
        dlSeg.append(dlSegment(pic, model))
    showImages(myPics + clSeg + dlSeg, 3)


    # Fish out of the water
    model = segModel()
    fg = getImages("./my_data/diffBg/fg")[0]
    bg = getImages("./my_data/diffBg/bg")[0]
    classMod = classModel() # Classification DL model
    pred = dlClassify(fg, classMod) # Deep Learning classify
    print(pred)
    fg_seg = dlSegment(fg, model)
    showImages([fg_seg])

    kang = changeBg(fg_seg, bg)
    showImages([kang])
    newPred = dlClassify(kang, classMod)
    cv2.imwrite("../output/Kangaroo_in_autria.jpg", cv2.cvtColor(kang, cv2.COLOR_RGB2BGR))
    print(newPred)
    print("All done!")
    if pred != newPred:
        print("Changing background fooled the neural net")


if __name__ == "__main__":

    main()