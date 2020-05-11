import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from sklearn.metrics import confusion_matrix
from network import Network
import time

###################################################################################################################################
#code reference:                                                                                                                  #
# code for training and test : https://github.com/JYPark09/SENet-PyTorch                                  #
# confusion matrix plot: https://deeplizard.com/learn/video/0LhiS6yu2qQ                                                           #
# visualization of misclassified samples: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_GAN.py#
###################################################################################################################################

###Initialization settings, global variable declarations.#####
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4
Train_acc = []
Train_loss = []
Test_acc = []
Test_loss = []
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

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    ###Loading using Mnist data set, data enhancement####
    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    #####Set the Batch size, and divide the training set and verification set.###
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,pin_memory=True)

    ###Model and model hyperparameter setting.#####
    net = Network(3, 128, 10, 10).to(device)
    ACE = nn.CrossEntropyLoss().to(device)
    opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)

    #####Accuracy curve drawing function#####
    def plot_acc_curves(array1, array2):
        plt.figure(figsize=(10, 10))
        x = np.linspace(1, EPOCHS, EPOCHS, endpoint=True)
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
        x = np.linspace(1, EPOCHS, EPOCHS, endpoint=True)
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
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

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
            all_preds = torch.cat((all_preds, preds), dim=0)
        return all_preds

    ####Visualization of misclassified images#####
    def plot_misclf_imgs(candidates, gts_np, preds_np, classes):
        size_figure_grid = 5  # a grid of 5 by 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(20, 20))

        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5 * 5):  # we have 25 grids to traverse
            i = k // 5
            j = k % 5
            idx = candidates[k]
            img = test_dataset[idx][0].numpy()
            img = img[0]
            ax[i, j].imshow((img), cmap='gray')  # draw the new one.
            ax[i, j].set_title("Label:" + str(classes[gts_np[idx]]), loc='left')
            ax[i, j].set_title("Predict:" + str(classes[preds_np[idx]]), loc='right')

        plt.savefig("misclf_imgs")
        plt.clf()

    ####Model training and model evaluation###
    best_acc = 0.0
    since = time.time()
    for epoch in range(1, EPOCHS + 1):
        print('[Epoch %d]' % epoch)

        train_loss = 0
        train_correct, train_total = 0, 0

        start_point = time.time()

        # for inputs, labels in train_loader:
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            opt.zero_grad()

            preds = net(inputs)

            loss = ACE(preds, labels)
            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        Train_acc.append(train_correct / train_total)
        Train_loss.append(train_loss / len(train_loader))
        print('train-acc : %.4f%% train-loss : %.5f' % (
        100 * train_correct / train_total, train_loss / len(train_loader)))
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0, 0

        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                preds = net(inputs)

                test_loss += ACE(preds, labels).item()

                test_correct += (preds.argmax(dim=1) == labels).sum().item()
                test_total += len(preds)
        test_acc = test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), './SENet_cifr10_40.pth')
        Test_acc.append(test_correct / test_total)
        Test_loss.append(test_loss / len(test_loader))
        print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader)))

        # torch.save(net.state_dict(), './checkpoint-%04d.bin' % epoch)
    time_elapsed = time.time() - since

    ####Get an array of related curves###
    test_preds = get_all_preds(net, test_loader).cpu()
    gts = test_dataset.targets
    preds = test_preds.argmax(dim=1)
    gts_np = np.array(gts)
    preds_np = np.array(preds)
    mis_idxes = list(np.where(gts_np != preds_np)[0])
    candidates = random.sample(mis_idxes, 25)
    cm = confusion_matrix(test_dataset.targets, test_preds.argmax(dim=1))

    ###Call functions for drawing curves, confusion matrix, misclassification###
    plot_confusion_matrix(cm, classes)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    plot_acc_curves(Train_acc, Test_acc)
    plot_loss_curves(Train_loss, Test_loss)
    plot_misclf_imgs(candidates, gts_np, preds_np, classes)

    ####Save the accuracy and loss results####
    Train_acc_np = np.array(Train_acc)
    Train_loss_np = np.array(Train_loss)
    Test_acc_np = np.array(Test_acc)
    Test_loss_np = np.array(Test_loss)
    np.savetxt("Train_acc.txt", Train_acc_np)
    np.savetxt("Train_loss.txt", Train_loss_np)
    np.savetxt("Test_acc.txt", Test_acc_np)
    np.savetxt("Test_loss.txt", Test_loss_np)
    print('Finished Training')

if __name__ == '__main__':
    main()
