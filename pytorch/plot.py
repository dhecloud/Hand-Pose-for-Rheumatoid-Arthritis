import matplotlib.pyplot as plt
import numpy as np

PATH_TO_LOSS_FILE ='experiments/exp_g_all/'

def save_plt(array, name):
    if 'train_loss' in name:
        plt.plot(array,color='red', label='train loss')
    else:
        plt.plot(array,color='blue', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('name')
    plt.legend()
    plt.savefig(name+'.png')

train_loss = np.loadtxt(PATH_TO_LOSS_FILE+"train_loss.out")
save_plt(train_loss, PATH_TO_LOSS_FILE+"train_loss")


val_loss = np.loadtxt(PATH_TO_LOSS_FILE+"val_loss.out")
save_plt(val_loss, PATH_TO_LOSS_FILE+"val_loss")
