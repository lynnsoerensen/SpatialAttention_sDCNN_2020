"""
Here, the firing rates for in and outside the focus of attention are obtained for the different attentional mechanisms and the neutral model.


"""

import joblib
import numpy as np
from asn.utils import load_pickle
from asn.evaluation.generators import coco_squared
from keras.models import model_from_json
from asn.layers.training import ASNTransfer
from keras import optimizers
from asn.conversion.convert import convert_model
from sklearn.model_selection import StratifiedShuffleSplit
from asn.attention.attn_param import set_attn_param
from keras.metrics import top_k_categorical_accuracy
from scipy.spatial import distance

from asn.evaluation.metrics import top_3_accuracy, top_2_accuracy, tp_rate, fp_rate

from keras.models import Model
from asn.evaluation.generators import TimeReplication_onset
from asn.evaluation.wrappers import predict_generator
from asn.layers.test import ASN_2D

# %% Settings
new_tracker = False
make_split = False
compute_dataset = True
compute_grayBaseline = False
compute_averageBaseline = True

#%%
# sess = tf.InteractiveSession()
# K.set_session(sess)
np.random.seed(3)

# %% load in the dataset:
setting = 'FSG_Precision_v0'

path = '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/BaselinePerformance_v0'
dataset = '1_street'  # ,'2_food']

img_dir_test = '/mnt/Googolplex/coco/images/' + dataset + '/val2017_single_radial/'
dataset_test = load_pickle(
    '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/Dataset/' + dataset + '/val2017_single_radial.pickle')
features = load_pickle(
    '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/Dataset/' + dataset + '/Features_single_radial.pickle')
targetCentres = [features['targetCentreoM'][s] for s in np.where(features['selection_1'])[0]]

images = [features['x_ids'][s] for s in np.where(features['selection_1'])[0]]
targets = [features['y'][s] for s in np.where(features['selection_1'])[0]]
areas = [features['targetArea'][s] for s in np.where(features['selection_1'])[0]]

r = np.round(np.sqrt(np.mean(np.array(areas)) / np.pi) * 224)

# Make sure that only the eligible target will be evaluated by including the selection filter
gen_test = coco_squared(dataset_test, batch_size=np.sum(features['selection_1']), img_dir=img_dir_test,
                        selection=features['selection_1'], shuffle=False)
(x, y) = next(gen_test)
if make_split == True:
    # make a class-balanced random split:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    split.get_n_splits(x, y)

    for train_index, test_index in split.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # ax_train, ax_test = targetCentres[train_index],  targetCentres[test_index]
else:
    train_index = np.arange(len(x))
    x_train = x
    y_train = y
# Centers for attention
Ax = np.zeros((len(train_index), 2))
Ax_invalid = np.zeros((len(train_index), 2))

train_idx_shuffle = np.arange(len(targetCentres))
np.random.shuffle(train_idx_shuffle)

for t in range(len(train_index)):
    Ax[t, 0] = targetCentres[train_index[t]][1]
    Ax[t, 1] = targetCentres[train_index[t]][0]

# Determine the typical range of object locations, such that extreme locations like 0.01, 0.99 will be avoided.
Ax_ranges = np.zeros((2,2))
Ax_ranges[0,:] = np.min(Ax,axis=0)
Ax_ranges[1,:] = np.max(Ax,axis=0)

for t in range(len(train_index)):
    Ax_invalid[t, 0] = targetCentres[train_idx_shuffle[t]][1]
    Ax_invalid[t, 1] = targetCentres[train_idx_shuffle[t]][0]

    while distance.euclidean(Ax[t,:], Ax_invalid[t,:]) < 0.5:
        Ax_invalid[t, 0] =np.random.uniform(Ax_ranges[0,0], Ax_ranges[1,0])
        Ax_invalid[t, 1]= np.random.uniform(Ax_ranges[0,1], Ax_ranges[1, 1])

test_locations = False
if test_locations == True:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(Ax[:,0], Ax[:, 1], color = 'red', label='valid')
    plt.scatter(Ax_invalid[:, 0], Ax[:, 1], color='blue', label='invalid')
    plt.xlim([0,1])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('/mnt/Googolplex/PycharmProjects/SpatialAttention_asn/Misc/CueLocations.png')

    distances = np.zeros(len(Ax))
    for t in range(len(Ax)):
        distances[t] = distance.euclidean(Ax[t,:], Ax_invalid[t,:])

test_images = False
if test_images == True:
    import matplotlib.pyplot as plt

    idx = 0#np.random.randint(len(x_train))
    fig, ax = plt.subplots()
    ax.imshow(x_train[idx, :, :, :])
    circle = plt.Circle((Ax[idx, 1] * 224, Ax[idx, 0] * 224), r, alpha=0.2)
    ax.add_artist(circle)
    ax.scatter(Ax[idx, 1] * 224, Ax[idx, 0] * 224, marker='o', c='r')
    ax.scatter((Ax_invalid[idx, 1]) * 224, (Ax_invalid[idx, 0]) * 224, marker='o')
    ax.set_title(y_train[idx])
    fig.savefig('/mnt/Googolplex/PycharmProjects/SpatialAttention_asn/Misc/TESTImage.png')

if compute_averageBaseline == True:
    img_dir_train = '/mnt/Googolplex/coco/images/' + dataset + '/train2017_multi_radial/'

    dataset_base = load_pickle(
        '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/Dataset/' + dataset + '/train2017_multi_radial.pickle')

    gen = coco_squared(dataset_base, batch_size=25, img_dir=img_dir_train)
    num_imgs = 16
    avg_imgs = np.zeros((num_imgs, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    for i in range(num_imgs):
        (x_base, y_base) = next(gen)
        avg_imgs[i] = np.mean(x_base, axis=0)

    del dataset_base, x_base, y_base, gen

#%%
np.random.seed(3)
# clear the memory
del dataset_test, features

#%%
tracker_path = '/mnt/Googolplex/PycharmProjects/SpatialAttention_asn/ModelAnalysis/'
tracker_name = 'EvokedPotentials.pkl'
if new_tracker == True:
    tracker = {}
    tracker['dataset'] = dataset
    tracker['Ax'] = {}
    tracker['Ax']['valid'] = Ax
    tracker['Ax']['invalid'] = Ax_invalid
else:
    tracker = joblib.load(tracker_path + tracker_name)

# %%  load in the model:
# some model params
optimizer = 'Adam'
optimizer_func = eval('optimizers.' + optimizer)
lr = 1e-4
weight_path = '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/FinetuneResnet18/Resnet_v0/' + dataset + '/weights_Adam_constant_0.0001.h5'

json_file = open(
    "/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/FinetuneResnet18/Resnet_v0/" + dataset + "/model.json",
    'r')

loaded_model_json = json_file.read()
json_file.close()

modelFT = model_from_json(loaded_model_json, custom_objects={'ASNTransfer': ASNTransfer})
modelFT.load_weights(weight_path)

modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                metrics=['binary_accuracy', 'categorical_accuracy', top_2_accuracy, top_3_accuracy])


#%% Prepare the spiking model params
time_steps = 750
start_eval = 250
batch_size = 2

empty = set_attn_param()
mf = 0.45
tracker['mf'] = mf


#%% and the attn model params
AxWidths = np.array([40])
layer_idx_att = np.array([3, 7, 11, 16, 20, 25, 29, 34])
stride_sizes = np.array([4, 4, 8, 8, 16, 16, 32, 32])
blocks = np.arange(1, 9)
block_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8])

successive = {}
for b in range(1, len(blocks) + 1):
    successive[b] = layer_idx_att[block_idx <= b]
b_idx = 8
a_idx = 0
l_att = successive[b_idx]

stride_current = {}
AxWidths_current = {}
for layer in range(len(l_att)):
    stride_current[layer] = stride_sizes[layer_idx_att == l_att[layer]]
    AxWidths_current[layer] = np.array(np.ceil(AxWidths / stride_current[layer]), dtype=int)


#%% Params for the spike train extraction
layer_idx = [25]#, 37]#34, 37]#, 25, 34]
conds = ['neutral', 'P-0_I-0_O-0.3','P-1_I-0_O-0.3','P-1_I-0_O-0', 'P-0_I-0.15_O-0', 'P-1_I-0.15_O-0','P-0_I-0.05_O-0.2', 'P-1_I-0.05_O-0.2']# 'P-1_I-0.15' 'P-1_I-0_O-0'] # Todo
#conds = ['P-0_I-0.05_O-0.2', 'P-1_I-0.05_O-0.2']
#%% Loop over the different models
for cond in conds:
    print(cond)
    # Make the models
    if cond == 'neutral':
        # without attention
        spikingModel, model_match = convert_model(modelFT, time_steps, {1: empty}, h_weightscaling=False,
                                             mf=mf, spike_counting=False)

        spikingModel.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    else:
        # Read-out the attentional settings
        splits = cond.split('_')
        precision_mode = bool(int(splits[0].split('-')[1]))
        gain_input = float(splits[1].split('-')[1])
        gain_output = float(splits[2].split('-')[1])

        attn_param = {}
        for layer in range(len(l_att)):
            attn_param[layer] = set_attn_param(layer_idx=l_att[layer],
                                               AxWidth1=AxWidths_current[layer][a_idx],
                                               AxWidth2=AxWidths_current[layer][a_idx],
                                               InputGain=gain_input, OutputGain=gain_output,
                                               Precision=precision_mode)
        # with attention
        spikingModel_att, model_match_att = convert_model(modelFT, time_steps, attn_param, h_weightscaling=False,
                                                                         mf=mf, spike_counting=False)

        spikingModel_att.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                                                   metrics=['accuracy'])

    # Obtain the firing rates for a layer across all conditions.
    for l in layer_idx:
        # l= 34
        print(l)
        if l not in tracker:
            tracker[l] = {}

        s = 16  # should be even.
        samples = np.arange(0, len(x_train), s)

        if cond == 'neutral':
            if cond not in tracker[l]:
                tracker[l][cond] = {}

            #model_shortened = Model(inputs=spikingModel.inputs, outputs=spikingModel.layers[l].output)

            readOut = ASN_2D(filters=spikingModel.layers[l].output.shape[-1].value, last_layer=True, mf=mf,
                                 kernel_initializer='ones',
                                 bias_initializer='zeros', h_scaling=True)(spikingModel.layers[l].output)
            model_shortened = Model(inputs=spikingModel.inputs, outputs=readOut)

            # Replace 1-weights with the identity matrix between the channel dimensions, such that every channel
            # is only connected to itself in the next layer for the readout.
            weights = model_shortened.layers[-1].get_weights()
            weights[0] = np.identity(weights[0].shape[-1]).reshape(weights[0].shape)
            model_shortened.layers[-1].set_weights(weights)  # Assign new weights
            if compute_grayBaseline == True:
                # Determine the baseline firing - without an image.
                grayImage = np.ones(x_train[:1].shape) * 0.5
                empty_y = np.zeros(y_train[:1].shape)
                baselinePredictions = predict_generator(model_shortened, grayImage,
                                                            empty_y,
                                                            time_steps,
                                                            batch_size,
                                                            generator=TimeReplication_onset,
                                                            steps=1)

                # Pick the neurons at the center, because there is no object location.
                #locations_baseline = np.array(np.floor(np.array([0.5, 0.5]) * model_shortened.output_shape[2]),
                #                              dtype=int)
                tracker[l][cond]['evoked_baseline'] = np.sum(
                        baselinePredictions[0, :,:, :, :], axis=-1)
                #tracker[l][cond]['MUA_all_baseline'] = np.sum(
                #    np.sum(np.sum(baselinePredictions[0] > 0, axis=1), axis=1),
                #    axis=1)  # sum over all neurons in this layer

                del baselinePredictions
                #joblib.dump(tracker, tracker_path + tracker_name,
                #            compress=True)

            if compute_averageBaseline == True:
                # Determine the baseline firing - with an average image.

                empty_y = np.zeros((avg_imgs.shape[0], y_train.shape[1]))
                baselinePredictions = predict_generator(model_shortened, avg_imgs,
                                                        empty_y,
                                                        time_steps,
                                                        batch_size,
                                                        generator=TimeReplication_onset,
                                                        steps=avg_imgs.shape[0]//batch_size)

                # Pick the neurons at the center, because there is no object location.
                # locations_baseline = np.array(np.floor(np.array([0.5, 0.5]) * model_shortened.output_shape[2]),
                #                              dtype=int)
                tracker[l][cond]['evoked_baseline_average'] = np.sum(
                    baselinePredictions, axis=-1)
                # tracker[l][cond]['MUA_all_baseline'] = np.sum(
                #    np.sum(np.sum(baselinePredictions[0] > 0, axis=1), axis=1),
                #    axis=1)  # sum over all neurons in this layer

                del baselinePredictions

            if compute_dataset == True:

                evoked = np.full((len(x_train), time_steps), np.nan)
                evoked_all = np.full((len(x_train), time_steps), np.nan)

                # This assumes that the activations are square:
                locations = np.array(np.floor(Ax * model_shortened.output_shape[2]), dtype=int)
                if 'locations' not in tracker[l]:
                    tracker[l]['locations'] = locations
                # Process the dataset in batches because otherwise, the memory requirements are too much
                print(cond)
                for n in range(len(samples)):
                    print(n)
                    if n == len(samples) - 1:
                        idx = np.arange(samples[n], len(x_train))
                    else:
                        idx = np.arange(samples[n], samples[n + 1])

                    spikingPredictions = predict_generator(model_shortened, x_train[idx],
                                                           y_train[idx],
                                                           time_steps,
                                                           batch_size,
                                                           generator=TimeReplication_onset,
                                                           steps=(len(idx) + 1) // batch_size)

                    for id_n in range(len(idx)):
                        # np.where(spikingPredictions[id_n] ==np.unique(spikingPredictions[id_n])[1])
                        # np.unique(spikingPredictions[id_n])[1]

                        # Extract the relevant neurons
                        neurons = spikingPredictions[id_n, :, locations[idx[id_n], 0], locations[idx[id_n], 1], :]

                        # and compute and store the MUA
                        evoked[idx[id_n]] = np.sum(neurons, axis=1)  # sum over feature maps
                        evoked_all[idx[id_n]] = np.sum(
                            np.sum(np.sum(spikingPredictions[id_n], axis=1), axis=1),
                            axis=1)  # sum over all neurons in this layer

                    del spikingPredictions

                # Write it back to the tracker
                tracker[l][cond]['evoked'] = evoked
                tracker[l][cond]['evoked_all'] = evoked_all

                del evoked_all, evoked

            joblib.dump(tracker, tracker_path + tracker_name,
                            compress=True)

            del model_shortened

        else:
            if cond not in tracker[l]:
                tracker[l][cond] = {}

            #if spikingModel_att.layers[l + 1].name.startswith('input'):
            #    layer_interest = spikingModel_att.layers[l]
            #elif spikingModel_att.layers[l + 1].name.startswith('time_distributed'):
            #    layer_interest = spikingModel_att.layers[l]
            #else:
            #    layer_interest = spikingModel_att.layers[l + 1]

            #model_shortened = Model(inputs=spikingModel_att.inputs, outputs=layer_interest.output)

            if spikingModel_att.layers[l + 1].name.startswith('input'):
                layer_interest = spikingModel_att.layers[l]
            elif spikingModel_att.layers[l + 1].name.startswith('time_distributed'):
                layer_interest = spikingModel_att.layers[l]
            else:
                layer_interest = spikingModel_att.layers[l + 1]

            # Place a spike readout layer
            readOut = ASN_2D(filters=layer_interest.output.shape[-1].value, last_layer=True, mf=mf,
                             kernel_initializer='ones',
                             bias_initializer='zeros', h_scaling=True)(layer_interest.output)

            model_shortened = Model(inputs=spikingModel_att.inputs, outputs=readOut)

            # Replace 1-weights with the identity matrix between the channel dimensions, such that every channel
            # is only connected to itself in the next layer
            weights = model_shortened.layers[-1].get_weights()
            weights[0] = np.identity(weights[0].shape[-1]).reshape(weights[0].shape)
            model_shortened.layers[-1].set_weights(weights)  # Assign new weights

            # This assumes that the activations are square:
            locations = np.array(np.floor(Ax * model_shortened.output_shape[2]), dtype=int)
            if 'locations' not in tracker[l]:
                tracker[l]['locations'] = locations

            for att in ['valid', 'invalid']:
                if att not in tracker[l][cond]:
                    tracker[l][cond][att] = {}

                if compute_grayBaseline == True:
                # Determine the baseline firing - without an image.
                    grayImage = np.ones(x_train[:1].shape) * 0.5
                    empty_y = np.zeros(y_train[:1].shape)

                    if att == 'valid':
                        # place the kernel in the center
                        att_loc_baseline = np.array([0.5, 0.5])[np.newaxis, :]
                    elif att == 'invalid':
                        # place the kernel in the periphery
                        att_loc_baseline = np.array([0.1, 0.2])[np.newaxis, :]
                    else:
                        raise ValueError('This cue condition is not implemented.')

                    baselinePredictions = predict_generator(model_shortened,
                                                            [grayImage, att_loc_baseline],
                                                            empty_y,
                                                            time_steps,
                                                            batch_size,
                                                            generator=TimeReplication_onset,
                                                            steps=1)

                    # Pick the neurons at the center, because there is no object location.
                    #locations_baseline = np.array(np.floor(np.array([0.5, 0.5]) * model_shortened.output_shape[2]),
                    #                              dtype=int)
                    #tracker[l][cond][att]['MUA_baseline'] = np.sum(
                    #    baselinePredictions[0, :, locations_baseline[0], locations_baseline[1], :] > 0,
                    #    axis=1)
                    #tracker[l][cond][att]['MUA_all_baseline'] = np.sum(
                    #    np.sum(np.sum(baselinePredictions[0] > 0, axis=1), axis=1),
                    #    axis=1)  # sum over all neurons in this layer

                    tracker[l][cond][att]['evoked_baseline'] = np.sum(
                        baselinePredictions[0, :, :, :, :], axis=-1)

                    del baselinePredictions

                if compute_averageBaseline == True:
                    # Determine the baseline firing - with an average image.

                    empty_y = np.zeros((avg_imgs.shape[0], y_train.shape[1]))

                    if att == 'valid':
                        # place the kernel in the center
                        att_loc_baseline = np.array([0.5, 0.5])[np.newaxis, :]
                    elif att == 'invalid':
                        # place the kernel in the periphery
                        att_loc_baseline = np.array([0.1, 0.2])[np.newaxis, :]
                    else:
                        raise ValueError('This cue condition is not implemented.')

                    baselinePredictions = predict_generator(model_shortened,
                                                            [avg_imgs,att_loc_baseline],
                                                            empty_y,
                                                            time_steps,
                                                            batch_size,
                                                            generator=TimeReplication_onset,
                                                            steps=1)

                    # Pick the neurons at the center, because there is no object location.
                    # locations_baseline = np.array(np.floor(np.array([0.5, 0.5]) * model_shortened.output_shape[2]),
                    #                              dtype=int)
                    tracker[l][cond][att]['evoked_baseline_average'] = np.sum(
                        baselinePredictions, axis=-1)
                    # tracker[l][cond]['MUA_all_baseline'] = np.sum(
                    #    np.sum(np.sum(baselinePredictions[0] > 0, axis=1), axis=1),
                    #    axis=1)  # sum over all neurons in this layer

                    del baselinePredictions



                if compute_dataset == True:
                    evoked = np.full((len(x_train), time_steps), np.nan)
                    evoked_all = np.full((len(x_train), time_steps), np.nan)
                    print(cond + ' ' + att)
                    for n in range(len(samples)):
                        print(n)
                        if n == len(samples) - 1:
                            idx = np.arange(samples[n], len(x_train))
                        else:
                            idx = np.arange(samples[n], samples[n + 1])

                        if att == 'valid':
                            cue = Ax
                        elif att == 'invalid':
                            cue = Ax_invalid
                        else:
                            raise ValueError('This cue condition is not implemented.')

                        spikingPredictions = predict_generator(model_shortened,
                                                               [x_train[idx], cue[idx]],
                                                               y_train[idx],
                                                               time_steps,
                                                               batch_size,
                                                               generator=TimeReplication_onset,
                                                               steps=(len(idx) + 1) // batch_size)

                        for id_n in range(len(idx)):
                            # np.where(spikingPredictions[id_n] ==np.unique(spikingPredictions[id_n])[1])
                            # np.unique(spikingPredictions[id_n])[1]

                            # Extract the relevant neurons
                            neurons = spikingPredictions[id_n, :, locations[idx[id_n], 0], locations[idx[id_n], 1], :]

                            # and compute and store the MUA
                            evoked[idx[id_n]] = np.sum(neurons, axis=1)  # sum over feature maps
                            evoked_all[idx[id_n]] = np.sum(
                                np.sum(np.sum(spikingPredictions[id_n], axis=1), axis=1),
                                axis=1)  # sum over all neurons in this layer

                        del spikingPredictions

                    # Write it back to the tracker
                    tracker[l][cond][att]['evoked'] = evoked
                    tracker[l][cond][att]['evoked_all'] = evoked_all

                    joblib.dump(tracker, tracker_path + tracker_name,
                                compress=True)

                    del evoked, evoked_all

            del model_shortened


"""
z = 16
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.arange(time_steps), mua_centre[:z,:].T, c = 'gray' )
plt.plot(np.arange(time_steps), np.mean(mua_centre[:z,:], axis=0), c = 'red' )
plt.savefig(tracker_path + 'Test_FR_centre.png')

plt.figure()
plt.plot(np.arange(time_steps), mua_peri[:z,:].T, c = 'gray' )
plt.plot(np.arange(time_steps), np.mean(mua_peri[:z,:], axis=0), c = 'red' )
plt.savefig(tracker_path + 'Test_FR_peri.png')

plt.figure()
plt.plot(np.arange(time_steps), np.mean(mua_centre[:z,:], axis=0), c = 'blue', label='centre')
plt.plot(np.arange(time_steps), np.mean(mua_peri[:z,:], axis=0), c = 'red', label='periphery')
plt.legend()
plt.savefig(tracker_path + 'Test_FR_comp.png')
"""


