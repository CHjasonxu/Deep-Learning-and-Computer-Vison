import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import time
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import itertools
import random
from model import vgg

###################################################################################################################################
#code reference:                                                                                                                  #
# code for training and test : https://github.com/WZMIAOMIAO/deep-learning-for-image-processing                                   #
# confusion matrix plot: https://deeplizard.com/learn/video/0LhiS6yu2qQ                                                           #
# visualization of misclassified samples: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_GAN.py#
###################################################################################################################################

###Initialization settings, global variable declarations.#####
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
nepoch=100
train_acc = []
test_acc = []
train_loss = []
test_loss = []

###Loading using Mnist data set, data enhancement####
# classes = (
#     '0',
#     '1',
#     '2',
#     '3',
#     '4',
#     '5',
#     '6',
#     '7',
#     '8',
#     '9'
# )
# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.50,), (0.50,))]),
#     "val": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.50,), (0.50,))])}

# train_dataset = torchvision.datasets.MNIST(root='../data',train=True,
#                                         download=True, transform=data_transform["train"])
# validate_dataset = torchvision.datasets.MNIST(root='../data',train=False,
#                                        download=True,transform=data_transform["val"])

###Loading using Mnist data set, data enhancement####
classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_dataset = torchvision.datasets.CIFAR10(root='../data',train=True,
                                        download=True, transform=data_transform["train"])
validate_dataset = torchvision.datasets.CIFAR10(root='../data',train=False,
                                       download=True,transform=data_transform["val"])


#####Set the Batch size, and divide the training set and verification set.###

batch_size = 64
train_num = len(train_dataset)
val_num = len(validate_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4,pin_memory=True)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4,pin_memory=True)

#####Accuracy curve drawing function#####
def plot_acc_curves(array1, array2):
    plt.figure(figsize=(10, 10))
    x = np.linspace(1, nepoch, nepoch, endpoint=True)
    plt.plot(x, array1, color='r', label='Train_accuracy')
    plt.plot(x, array2, color='b', label='Test_accuracy')
    plt.legend()
    plt.title('accuracy of train and test sets in different epoch')

    plt.xlabel('epoch')
    plt.ylabel('accuracy: ')
    plt.savefig("acc_curves")
    plt.show()
    plt.clf()

#####Loss curve drawing function#####
def plot_loss_curves(array1, array2):
    plt.figure(figsize=(10, 10))
    x = np.linspace(1, nepoch, nepoch, endpoint=True)
    plt.plot(x, array1, color='r', label='Train_loss')
    plt.plot(x, array2, color='b', label='Test_loss')
    plt.legend()
    plt.title('loss of train and test sets in different epoch')

    plt.xlabel('epoch')
    plt.ylabel('loss: ')
    plt.savefig("loss_curves")
    plt.show()
    plt.clf()

#####Confusion Matrix drawing function#####
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix")
    plt.clf()

###Get prediction result function###
@torch.no_grad()
def get_all_preds(model, loader):

    all_preds = torch.tensor([]).to(device)
    model.to(device)
    for batch in loader:
        images, labels = batch
        preds = model(images.to(device))
        all_preds = torch.cat((all_preds, preds),dim=0)
    return all_preds

####Visualization of misclassified images#####
def plot_misclf_imgs(candidates,gts_np,preds_np,classes):
    size_figure_grid = 5  # a grid of 5 by 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(20, 20))

    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):  # we have 25 grids to traverse
        i = k // 5
        j = k % 5
        idx = candidates[k]
        img = validate_dataset[idx][0].numpy()
        img = img[0]
        ax[i, j].imshow((img), cmap='gray')  # draw the new one.
        ax[i, j].set_title("Label:"+str(classes[gts_np[idx]]), loc='left')
        ax[i, j].set_title("Predict:"+str(classes[preds_np[idx]]), loc='right')

    plt.savefig("misclf_imgs")
    plt.clf()


###Model and model hyperparameter setting.#####
model_name = "vgg16"
net = vgg(model_name=model_name, class_num=10, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4,factor=0.1)

####Model training and model evaluation###
best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)
since = time.time()
for epoch in range(nepoch):
    # train
    net.train()
    running_loss = 0.0
    running_corrects = 0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        _, predict_y = torch.max(logits, dim=1)
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        running_corrects += (predict_y == labels.to(device)).sum().item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()
    accurate_train = running_corrects / train_num
    train_loss.append(running_loss / len(train_loader))
    train_acc.append(accurate_train)
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    Loss_val = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))  # eval model only have last output layer
            loss_val = loss_function(outputs, test_labels.to(device))
            Loss_val += loss_val.item()
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        test_acc.append(val_accurate)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        test_loss.append(Loss_val / len(validate_loader))
    # scheduler.step(loss_val)
time_elapsed = time.time() - since

####Get an array of related curves###
test_preds = get_all_preds(net, validate_loader).cpu()
gts = validate_dataset.targets
preds = test_preds.argmax(dim=1)
gts_np = np.array(gts)
preds_np = np.array(preds)
mis_idxes = list(np.where(gts_np!= preds_np)[0])
candidates = random.sample(mis_idxes,25)
cm = confusion_matrix(validate_dataset.targets, test_preds.argmax(dim=1))

###Call functions for drawing curves, confusion matrix, misclassification###
plot_confusion_matrix(cm, classes)
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
plot_acc_curves(train_acc,test_acc)
plot_loss_curves(train_loss,test_loss)
plot_misclf_imgs(candidates,gts_np,preds_np,classes)

####Save the accuracy and loss results####
Train_acc_np = np.array(train_acc)
Train_loss_np = np.array(train_loss)
Test_acc_np = np.array(test_acc)
Test_loss_np = np.array(test_loss)
np.savetxt("Train_acc.txt", Train_acc_np)
np.savetxt("Train_loss.txt", Train_loss_np)
np.savetxt("Test_acc.txt", Test_acc_np)
np.savetxt("Test_loss.txt", Test_loss_np)
print('Finished Training')