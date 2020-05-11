import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(array1,array2,array3,array4,array5,array6,array7,array8,epoch):
    plt.figure(figsize=(10, 10))
    x=np.linspace(1,epoch,epoch,endpoint=True)
    plt.plot(x,array1,color='r',label='resnet_relu_Train_loss')
    plt.plot(x,array2,color='r', label='resnet_relu_Test_loss',linestyle="--")
    plt.plot(x,array3,color='b',label='resnet_prelu_Train_loss')
    plt.plot(x,array4,color='b', label='resent_prelu_Test_loss',linestyle="--")
    plt.plot(x, array5, color='g', label='googlenet_Train_loss')
    plt.plot(x, array6, color='g', label='googlenet_Test_loss',linestyle="--")
    plt.plot(x, array7, color='c', label='senet_Train_loss')
    plt.plot(x, array8, color='c', label='senet_Test_loss',linestyle="--")
    plt.legend()
    plt.title('loss of train and test sets in different epoch')
    
    plt.xlabel('epoch')
    plt.ylabel('loss: ')
    fig = plt.gcf()
    plt.show()
    fig.savefig("Cifr10_loss_curves")
    plt.clf()

def plot_acc_curves(array1,array2,array3,array4,array5,array6,array7,array8,epoch):
    plt.figure(figsize=(10, 10))
    x=np.linspace(1,epoch,epoch,endpoint=True)
    plt.plot(x,array1,color='r',label='resnet_relu_Train_accuracy')
    plt.plot(x,array2,color='r', label='resnet_relu_Test_accuracy',linestyle="--")
    plt.plot(x,array3,color='b',label='resnet_prelu_Train_accuracy')
    plt.plot(x,array4,color='b', label='resent_prelu_Test_accuracy',linestyle="--")
    plt.plot(x, array5, color='g', label='googlenet_Train_accuracy')
    plt.plot(x, array6, color='g', label='googlenet_Test_accuracy',linestyle="--")
    plt.plot(x, array7, color='c', label='senet_Train_accuracy')
    plt.plot(x, array8, color='c', label='senet_Test_accuracy',linestyle="--")
    
    plt.legend()
    plt.title('accuracy of train and test sets in different epoch')
    
    plt.xlabel('epoch')
    plt.ylabel('accuracy: %')
    fig=plt.gcf()
    plt.show()
    fig.savefig("Cifr10_acc_curves")
    plt.clf()

epoch =50


r_all_train_acc = np.loadtxt("r_all_train_acc.txt")
r_all_val_acc = np.loadtxt("r_all_val_acc.txt")
r_all_train_loss = np.loadtxt("r_all_train_loss.txt")
r_all_val_loss = np.loadtxt("r_all_val_loss.txt")

r_p_train_acc = np.loadtxt("r_p_train_acc.txt")
r_p_val_acc = np.loadtxt("r_p_val_acc.txt")
r_p_train_loss = np.loadtxt("r_p_train_loss.txt")
r_p_val_loss = np.loadtxt("r_p_val_loss.txt")

g_all_train_acc = np.loadtxt("g_all_train_acc.txt")
g_all_val_acc = np.loadtxt("g_all_val_acc.txt")
g_all_train_loss = np.loadtxt("g_all_train_loss.txt")
g_all_val_loss = np.loadtxt("g_all_val_loss.txt")

s_all_train_acc = np.loadtxt("s_all_train_acc.txt")
s_all_val_acc = np.loadtxt("s_all_val_acc.txt")
s_all_train_loss = np.loadtxt("s_all_train_loss.txt")
s_all_val_loss = np.loadtxt("s_all_val_loss.txt")

plot_acc_curves(r_all_train_acc*100,r_all_val_acc*100,r_p_train_acc*100,r_p_val_acc*100,g_all_train_acc*100,g_all_val_acc*100,s_all_train_acc*100,s_all_val_acc*100,epoch)
plot_loss_curves(r_all_train_loss,r_all_val_loss,r_p_train_loss,r_p_val_loss,g_all_train_loss,g_all_val_loss,s_all_train_loss,s_all_val_loss,epoch)