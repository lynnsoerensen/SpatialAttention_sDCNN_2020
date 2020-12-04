import matplotlib.pyplot as plt
import numpy as np


def plot_training(history, path, title=None):

    epochs = len(history['loss'])
    if 'lr' in history:
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax = ax.flatten()
        ax[0].plot(np.arange(epochs), np.transpose([history['loss'], history['val_loss']]))
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('loss')
        ax[0].set_ylim([0, 3])
        ax[0].legend(['training', 'validation'])

        ax[1].plot(np.arange(epochs), np.transpose([history['acc'], history['val_acc']]))
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('accuracy')
        ax[1].set_ylim([0, 1])
        ax[1].legend(['training', 'validation'])

        ax[2].plot(np.arange(epochs), history['lr'])
        ax[2].set_xlabel('epochs')
        ax[2].set_ylabel('learning rate')
        # ax[2].set_ylim([0,1])
    else:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax = ax.flatten()
        ax[0].plot(np.arange(epochs), np.transpose([history['loss'], history['val_loss']]))
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('loss')
        ax[0].set_ylim([0, 3])

        ax[1].plot(np.arange(epochs), np.transpose([history['acc'], history['val_acc']]))
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('accuracy')
        ax[1].set_ylim([0, 1])
        fig.legend(['training', 'validation'])
    if title:
        fig.suptitle(title)
    fig.savefig(path)
