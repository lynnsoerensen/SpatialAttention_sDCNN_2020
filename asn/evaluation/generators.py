#Collection of Datagenerators for both testing and training
import numpy as np
import keras
from keras.datasets import mnist
#from scipy.misc import imresize
from keras.preprocessing import image


def TimeReplication(X_test, y_test, time_steps, batch_size=1):
    # Replicate the same image over multiple time steps and for given amount of images within a batch
    # You can call a generator object with next(gen_object)
    if isinstance(X_test,list):
        Ax = X_test[1]
        X_all = X_test[0]
    else:
        X_all = X_test

    counter = 0
    while X_all.shape[0] - counter*batch_size > 0:
        # Duplicate the given image across all time-points
        X_batch = X_all[counter * batch_size: (counter + 1) * batch_size, :, :, :]
        X = np.repeat(X_batch[:, np.newaxis, :, :, :], time_steps, axis=1)

        y_batch = y_test[counter * batch_size: (counter + 1) * batch_size, :]
        y = np.repeat(y_batch[:, np.newaxis, :], time_steps, axis=1)

        if isinstance(X_test, list):
            X = [X,Ax[counter * batch_size: (counter + 1) * batch_size,:]]

        counter = counter + 1
        yield X, y


def TimeReplication_onset(X_test, y_test, time_steps, batch_size=1, onset=100, offset=None, fill_value=0.5):
    # Replicate the same image over multiple time steps and for given amount of images within a batch
    # You can call a generator object with next(gen_object)

    # This is an adaptation from asn.generators.TimeReplication
    # 25.03.2020
    if isinstance(X_test, list):
        Ax = X_test[1]
        X_all = X_test[0]
    else:
        X_all = X_test

    counter = 0

    while X_all.shape[0] - counter * batch_size > 0:
        # Duplicate the given image across all time-points
        X_batch = X_all[counter * batch_size: (counter + 1) * batch_size, :, :, :]
        X = np.repeat(X_batch[:, np.newaxis, :, :, :], time_steps, axis=1)

        y_batch = y_test[counter * batch_size: (counter + 1) * batch_size, :]
        y = np.repeat(y_batch[:, np.newaxis, :], time_steps, axis=1)

        if onset is not None:
            X[:, :onset, :, :, :] = fill_value
            y[:, :onset, :] = 0

        if offset is not None:
            X[:, offset:, :, :, :] = fill_value
            y[:, offset:, :] = 0

        if isinstance(X_test, list):
            X = [X, Ax[counter * batch_size: (counter + 1) * batch_size, :]]

        counter = counter + 1
        yield X, y


def coco_squared(dataset, batch_size = 1, target_size = (224,224),
                 img_dir = '/mnt/Googolplex/coco/images/2_food/train2017_multi_radial/', shuffle=True, selection=None,
                 singleTarget=False, doubleOutput=False, location=False):

    """
    :param dataset: A dict containing at least two fields: x_ids & y
    :param batch_size: How many images to produce at every call
    :param target_size: What should the image dimensions?
    :param img_dir: Where are the images stored?
    :param shuffle: Whether to shuffle the dataset once it was looped over once.
    :param selection: Whether to use a subselection of the dataset
    :param singleTarget: If true, the generator will first create a bigger dataset with repeated images for multi-label cases
    :return: Pair of images and labels.
    """

    if selection is not None:
        if selection[0].dtype == 'bool':
            #  Convert bool to index
            selection = np.where(selection)[0]

        if type(dataset['x_ids'][0]) == str:
            # x_ids = [dataset['x_ids'][d] for d in range(len(dataset['x_ids'])) if selection[d]]

            x_ids = [dataset['x_ids'][s] for s in selection]
        else:
            x_ids = dataset['x_ids'][selection]

        y_ids =dataset['y'][selection]

        if 'annIds' in dataset:
            ann_ids = [dataset['annIds'][s] for s in selection]
        if location == True:
            targetCentres = [dataset['CoM'][s] for s in selection]
        if 'y_ann' in dataset:
            y_ann = [dataset['y_ann'][s] for s in selection]


    else:
        x_ids = dataset['x_ids']
        y_ids = dataset['y']
        if 'annIds' in dataset:
            ann_ids = dataset['annIds']

        if location == True:
            targetCentres = dataset['CoM']
        if 'y_ann' in dataset:
            y_ann = dataset['y_ann']

    if singleTarget== True:
        # Find the images with multiple labels and duplicate them so that every image only has one target,
        new_y_ids = []
        new_x_ids = []
        if location == True:
            new_targetCentres = []

        if 'y_ann' in dataset:
            for i in range(len(y_ann)): # loop over images
                for j, y in enumerate(y_ann[i]): # and annotations
                    new_x_ids.append(x_ids[i])
                    new_y_ids.append(y[0,:])
                    if location == True:
                        new_targetCentres.append(targetCentres[i][j])

        else:
            for i, y in enumerate(y_ids): # loop p
                idx = np.where(y == 1)
                for j in range(len(idx[0])):

                    label = np.zeros(len(y))
                    label[idx[0][j]] = 1
                    new_y_ids.append(label)
                    new_x_ids.append(x_ids[i])

        # Once this is done, shuffle!
        idx = np.arange(len(new_x_ids))
        np.random.shuffle(idx)
        x_ids = np.array(new_x_ids)[idx]
        y_ids = np.array(new_y_ids)[idx]
        if location == True:
            targetCentres = np.array(new_targetCentres)[idx]

    counter = 0
    while len(x_ids) - (counter+1) * batch_size >= 0:
        x = np.zeros((batch_size,) + target_size + (3,))
        if doubleOutput == True:
            y = np.zeros((batch_size,) + (y_ids.shape[1]*2,))
        else:
            y = np.zeros((batch_size,) + (y_ids.shape[1],))
            locs = np.zeros((batch_size,) + (2,))

        batch = np.arange(counter * (batch_size), (counter + 1) * (batch_size))

        for i in range(batch_size):
            if (type(x_ids[0]) == str):  #| (type(x_ids[0][0]) == str):
                id = x_ids[batch[i]]
            elif (dataset['dataType'] == 'val2017') & (type(x_ids[0]) != str):
                id = str(ann_ids[batch[i]][0]) + '_' + str(x_ids[batch[i]]).zfill(12) + '.jpg'
            else:
                id = str(x_ids[batch[i]]).zfill(12) + '.jpg'
            img = image.load_img(img_dir + id, target_size=target_size)
            img = image.img_to_array(img)
            img = img / 255
            x[i,] = np.expand_dims(img, axis=0)
            if doubleOutput == True:
                for j, k in enumerate(y_ids[batch[i]]):
                    if k == 1:
                        y[i, j*2] = 1 # even indices are yes
                    else:
                        y[i, (j * 2) + 1] = 1  # odd indices are no
            else:
                y[i,] = y_ids[batch[i]]
                if location == True:
                    locs[i,] = targetCentres[batch[i]]

        counter = counter + 1
        if len(x_ids) - (counter+1) * batch_size <= 0: # After very last iteration
            counter = 0
            if shuffle == True:
                idx = np.arange(len(x_ids))
                np.random.shuffle(idx)
                if type(x_ids[0]) == str:
                    x_ids = [x_ids[m] for m in idx]
                else:
                    x_ids = x_ids[idx]
                y_ids = y_ids[idx]
                if location == True:
                    targetCentres = targetCentres[idx]

        if location == True:
            yield [x,locs],y
        else:
            yield x,y


