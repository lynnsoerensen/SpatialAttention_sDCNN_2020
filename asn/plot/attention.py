import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.preprocessing import image
from asn.attention.attn_param import set_attn_param
from asn.attention.attend import attend_tf
import tensorflow as tf
import keras.backend as K


def plotCueLocations(img_path, output_path, valid_loc=None, invalid_loc=None):
    sns.set_context("poster")
    colors = sns.color_palette(sns.xkcd_palette([ "amber","windows blue", "greyish", "faded green", "dusty purple","orange"]))

    img = image.load_img(img_path)#,target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255

    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(img)
    ax.axis('off')
    if valid_loc is not None:
        ax.scatter(valid_loc[1] * img.shape[0],valid_loc[0] * img.shape[1],  s=500, c=colors[0], label='valid')
    if invalid_loc is not None:
        ax.scatter(invalid_loc[1] * img.shape[0],invalid_loc[0] * img.shape[1],  s=500, c=colors[1], label='invalid')

    fig.legend(frameon=False)
    plt.savefig(output_path)


def plotAttentionField(img_path, output_path, valid_loc):
    sns.set_context("poster")
    colors = sns.color_palette(
        sns.xkcd_palette(["amber", "windows blue", "greyish", "faded green", "dusty purple", "orange"]))

    valid_loc = np.array([valid_loc])
    sess = tf.InteractiveSession()
    K.set_session(sess)

    AxWidth = np.array([40])

    init = tf.global_variables_initializer()
    init.run()

    attn_param = set_attn_param(AxWidth1=AxWidth, AxWidth2=AxWidth, Precision=True)
    R = attend_tf([1, 224, 224, 3], 'channels_last', attn_param, Ax1=valid_loc[:, 0], Ax2=valid_loc[:, 1])

    R_mat = R.eval()
    fig, ax = plt.subplots(figsize=(8, 8))

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255

    ax.imshow(img)
    im = ax.imshow(R_mat[0, :, :, 0], plt.set_cmap('cividis'), alpha=0.6)
    ax.scatter(valid_loc[0, 1] * img.shape[0], valid_loc[0, 0] * img.shape[1], s=250, c=colors[0], label='valid')

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Change')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - 0.08, box.width, box.height * 1.1])
    #fig.suptitle('Attention field')
    sns.despine(bottom=True, left=True)
    plt.savefig(output_path)


