import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import cv2
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transform


class defaultCNN(nn.Module):
    """Generic CNN for SVHN dataset"""

    def __init__(self):
        """"CNN constructor"""
        super(defaultCNN, self).__init__()
        # Set CNN architecture
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Set FC architecture
        self.fc_layer = nn.Sequential(
            # Linea layer 1
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            # Lenear layer 2
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            # Linear layer 3
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward pass"""
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


def convert_to_imshow_format(image):
    """convert_to_imshow_format(image) -> image_formatted"""
    """Changes the image into a format fits imshow()"""
    # first convert back to [0,1] range from [-1,1] range - approximately...
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1,2,0)


def showDatasetImages(set, N=5):
    """showDatasetImages(set, N=5)"""
    """Show N images out of a given dataset"""
    # Get N images out of set and puts them in a list with labels
    trainloader = torch.utils.data.DataLoader(set,
                                              batch_size=N,
                                              shuffle=True)
    dataiter = iter(trainloader)
    images,labels= dataiter.next()
    # sets the labels
    classes = range(10)
    fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
    for idx, image in enumerate(images):
        axes[idx].imshow(convert_to_imshow_format(image))
        axes[idx].set_title(classes[labels[idx]])
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.show()


def getHyperParam():
    """getHyperParam() -> HP_dictionary"""
    """Get a Hyper Parameters dictionary"""
    dict = {'batch_size': 2**8,
            'learning_rate': 1e-4,
            'epochs': 10,
            'debug': False,
            'path': "./models/default_cnn.pth",
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}
    return dict


def saveModel(model, HP):
    """saveModel(model,HP)"""
    """Saves the current model with it's hyper parameters"""
    """Used only in debug mode"""
    if HP['debug']:
        print('==> Saving model ...')
        state = {
            'net': model.state_dict(),
            'epoch': HP['epochs'],
            'batch_size': HP['batch_size'],
            'learning_rate': HP['learning_rate']
        }
        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(state, HP['path'])


def calculate_accuracy(model, dataSet, device):
    """calculate_accuracy(model, data, device) -> model_acc , confusion_mat"""
    """Calculates the accuracy of the model towards the data"""
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    # Doesn't calculate back prog
    with torch.no_grad():
        for data in dataSet:
            # Get data
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) # Run through model
            _, predicted = torch.max(outputs.data, 1) # pick label
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1 # Correct prediction should be in the diagonal

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix


if __name__ == "__main__":
    # Set hyper parameters
    HPdict = getHyperParam()
    batch_size = HPdict['batch_size']
    learning_rate = HPdict['learning_rate']
    epochs = HPdict['epochs']
    device = HPdict['device']
    root = "./datasets"

    # Load Data sets
    trainSet = torchvision.datasets.SVHN(root, split='train', transform=transform.ToTensor(), target_transform=None, download=True)
    testsSet = torchvision.datasets.SVHN(root, split='test', transform=transform.ToTensor(), target_transform=None, download=True)
    if (HPdict['debug']):
        # for making training time much shorter
        ind = np.random.randint(1, len(testsSet), 8 * HPdict['batch_size'])
        trainSet = Subset(trainSet, ind)
        testsSet = Subset(testsSet, ind)
    trainLoad = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    testsLoad = DataLoader(testsSet, batch_size=batch_size, shuffle=False, num_workers=2)

    showDatasetImages(trainSet)

    # Set NN
    model = defaultCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    # Set a logging mechanism
    log = pd.DataFrame({}, columns=['Epoch', 'Batch', 'Loss', 'Epoch loss', 'Train acc', 'Test acc', 'Learning rate', 'Time'])

    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time() # start counter for epoch
        for i, data in enumerate(trainLoad):
            logBatch = {'Epoch': epoch, 'Batch': i, 'Learning rate': learning_rate}
            # get the inputs
            inputs, labels = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # log statistics
            running_loss += loss.data.item()
            logBatch['Loss'] = loss.tolist()
            log = log.append(logBatch, ignore_index=True)
        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainLoad)

        # Calculate training/test set accuracy of the existing model
        train_accuracy, _ = calculate_accuracy(model, trainLoad, device)
        test_accuracy, _ = calculate_accuracy(model, testsLoad, device)
        # log epoch
        log = log.append({'Epoch loss': running_loss,
                          'Train acc': train_accuracy,
                          'Test acc': test_accuracy,
                          'Time': time.time()-epoch_time},
                         ignore_index=True)
        # Decrease learning rate by the end of the epoch
        learning_rate = learning_rate * 0.85
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    # Test
    test_accuracy, confusion_matrix = calculate_accuracy(model, testsLoad, device)
    # Give statistics
    print(log)
    print("test accuracy: {:.3f}%".format(test_accuracy))
    log.interpolate().plot(y=['Loss', 'Epoch loss'])
    plt.title('Model loss')
    plt.xlabel('Batch number')
    plt.ylabel('loss')
    plt.show()
    log.interpolate().plot(y=['Train acc', 'Test acc'])
    plt.title("Model accuracy")
    plt.xlabel('Batch number')
    plt.ylabel('Accuracy [%}')
    plt.show()

    # plot confusion matrix
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    plt.yticks(range(10), classes)
    plt.xlabel('Predicted Category')
    plt.xticks(range(10), classes)
    plt.show()

    # for debug mode
    #saveModel(model, HPdict)
    #log.to_csv('./logError.csv')