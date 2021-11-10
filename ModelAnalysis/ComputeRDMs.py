"""
1. Here, the evoked or analog potentials are first obtained for the images contributing to the rdm.
2. Then these are stored in a numpy memory map.
3. After memory is cleared, the activations will be analysed with pdist to obtain the rdm.

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
from sklearn.metrics import roc_auc_score
from keras.models import Model
from asn.evaluation.generators import TimeReplication_onset
from asn.evaluation.wrappers import predict_generator
from asn.layers.test import ASN_2D
import time
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


def compute_rdm(activations, method='corr'):
    """
    :param activations: An m by n array of m original observations in an n-dimensional space (nImages x nUnits)
    :param method: correlation or ranked correlation.
    :return: representational dissimilarity matrix

    https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/rank_correlations/
    """
    if method == 'corr':
        rdm = pdist(activations, 'correlation')
    elif method == 'corr-rank':
        rdm = rankdata(pdist(activations, 'correlation'))
    else:
        raise ValueError('The specified method ' + method + 'is not implemented.')

    return rdm

dir_path = os.path.abspath('')
# %% Settings
new_tracker = False
make_selection = True
check_performance = False

#%%
# sess = tf.InteractiveSession()
# K.set_session(sess)
np.random.seed(3)

# %% load in the dataset:
setting = 'FSG_Precision_v0'

dataset = '1_street'

img_dir_test = dir_path + '/Datasets/' + dataset + '/images/val2017_single_radial/'
dataset_test = load_pickle(dir_path + '/Datasets/' + dataset + '/val2017_single_radial.pickle')
features = load_pickle(dir_path + '/Datasets/' + dataset + '/Features_single_radial.pickle')
targetCentres = [features['targetCentreoM'][s] for s in np.where(features['selection_1'])[0]]

images = [features['x_ids'][s] for s in np.where(features['selection_1'])[0]]
targets = [features['y'][s] for s in np.where(features['selection_1'])[0]]
areas = [features['targetArea'][s] for s in np.where(features['selection_1'])[0]]

r = np.round(np.sqrt(np.mean(np.array(areas)) / np.pi) * 224)

# Make sure that only the eligible target will be evaluated by including the selection filter
gen_test = coco_squared(dataset_test, batch_size=np.sum(features['selection_1']), img_dir=img_dir_test,
                        selection=features['selection_1'], shuffle=False)
(x, y) = next(gen_test)

category_order = ['person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck', 'stop sign', 'fire hydrant']

if make_selection == True:
    # pick the a random 50 images of every category
    num_imgs = 50
    for cat in range(y.shape[1]):
        cat_id = [dataset_test['categories'][c] == category_order[cat] for c in range(len(dataset_test['categories']))]
        imgs = np.where(y[:, cat_id])[0]
        np.random.shuffle(imgs)
        if cat == 0:
            train_index = imgs[:num_imgs]
        else:
            train_index = np.append(train_index, imgs[:num_imgs])

    x_train = x[train_index]
    y_train = y[train_index]
else:
    train_index = np.arange(len(x))
    x_train = x
    y_train = y

# Centers for attention
Ax = np.zeros((len(train_index), 2))
Ax_invalid = np.zeros((len(train_index), 2))
areas_sel = np.zeros(len(train_index))

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
        Ax_invalid[t, 0] = np.random.uniform(Ax_ranges[0,0], Ax_ranges[1,0])
        Ax_invalid[t, 1] = np.random.uniform(Ax_ranges[0,1], Ax_ranges[1, 1])



#%%
np.random.seed(3)
# clear the memory
del dataset_test, features


#%%
tracker_path = dir_path + '/ModelAnalysis/RSA/'
tracker_name = 'RDMs.pkl'
if new_tracker == True:
    tracker = {}
    tracker['dataset'] = dataset
    tracker['categories'] = category_order
else:
    tracker = joblib.load(tracker_path + tracker_name)
# %%  load in the model:
# some model params
optimizer = 'Adam'
optimizer_func = eval('optimizers.' + optimizer)
lr = 1e-4
weight_path = dir_path + '/ModelTraining/Finetune_coco/' + dataset + '/weights_Adam_constant_0.0001.h5'

json_file = open(dir_path + '/ModelTraining/Finetune_coco/' + dataset + "/model.json",'r')

loaded_model_json = json_file.read()
json_file.close()

modelFT = model_from_json(loaded_model_json, custom_objects={'ASNTransfer': ASNTransfer})
modelFT.load_weights(weight_path)

modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                metrics=['binary_accuracy', 'categorical_accuracy', top_2_accuracy, top_3_accuracy])
if check_performance == True:
    preds = modelFT.predict(x_train)
    tracker['AnalogPerformance'] = roc_auc_score(y_train,preds)

#%% Prepare the spiking model params
time_steps = 750
start_eval = 250
batch_size = 2

empty = set_attn_param()
mf = 0.45

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
layer_idx = [37]
l = 37
conds = ['P-0_I-0.15_O-0','analog', 'neutral', 'P-0_I-0_O-0.3', 'P-0.45_I-0_O-0']

rdm_path = '/mnt/Googolplex/PycharmProjects/SpatialAttention_asn/ModelAnalysis/RSA/'
data_path = '/mnt/Googleplex2/RSA_activations/'

#%% Loop over the different models
for cond in conds:
    print(cond)
    if 'RDMs' not in tracker:
        tracker['RDMs'] = {}
    # Make the models
    if cond == 'analog':
        print('Model already loaded.')
    elif cond == 'neutral':
        # without attention
        spikingModel, model_match = convert_model(modelFT, time_steps, {1: empty}, h_weightscaling=False,
                                             mf=mf, spike_counting=False)

        spikingModel.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    else:
        # Read-out the attentional settings
        splits = cond.split('_')
        precision_mode = float(splits[0].split('-')[1])
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

    if cond not in tracker['RDMs']:
        tracker['RDMs'][cond] = {}

    if 'corr' not in tracker['RDMs'][cond]:
        if cond == 'analog':
            # Obtain the analog prediction

            l_id = 65
            model_shortened = Model(input=modelFT.input, output=modelFT.layers[l_id].output)
            activations = model_shortened.predict(x_train)

            rdm = compute_rdm(activations.reshape(activations.shape[0], -1))


            tracker['RDMs'][cond]['corr'] = rdm


            joblib.dump(tracker, tracker_path + tracker_name, compress=True)

            del model_shortened, rdm

        else:

            s = 16  # should be even.
            samples = np.arange(0, len(x_train), s)

            if cond == 'neutral':
                data_name = data_path + cond + '_' + str(l) + '_data.mmap'

                # Place a spike readout layer
                readOut = ASN_2D(filters=spikingModel.layers[l].output.shape[-1].value, last_layer=True, mf=mf,
                                         kernel_initializer='ones',
                                         bias_initializer='zeros', h_scaling=True)(spikingModel.layers[l].output)
                model_shortened = Model(inputs=spikingModel.inputs, outputs=readOut)

                # Replace 1-weights with the identity matrix between the channel dimensions, such that every channel
                # is only connected to itself in the next layer for the readout.
                weights = model_shortened.layers[-1].get_weights()
                weights[0] = np.identity(weights[0].shape[-1]).reshape(weights[0].shape)
                model_shortened.layers[-1].set_weights(weights)  # Assign new weights

                data_shape = (len(x_train),) + model_shortened.output_shape[1:]
                if not os.path.isfile(data_name):
                    # Process the dataset in batches because otherwise, the memory requirements are too much if it's a sDCNN
                    # Preallocate the memory map
                    data_map = np.memmap(data_name, dtype='float32', shape=data_shape, mode='w+')
                    del data_map

                # Process the dataset in batches because otherwise, the memory requirements are too much

                for n in range(len(samples)):
                    print(n)
                    if n == len(samples)-1:
                        idx = np.arange(samples[n], len(x_train))
                    else:
                        idx = np.arange(samples[n],samples[n+1])

                    data_map = np.memmap(data_name, dtype='float32', shape=data_shape, mode='r+')
                    data_map[idx] = predict_generator(model_shortened, x_train[idx],
                                                                       y_train[idx],
                                                                       time_steps,
                                                                       batch_size,
                                                                       generator=TimeReplication_onset,
                                                                       steps=len(idx)//batch_size)

                    del data_map

                data_map = np.memmap(data_name, dtype='float32', shape=data_shape, mode='r+')

                rdm = np.zeros((data_shape[1], int(data_shape[0] * (data_shape[0] - 1) / 2)))

                print('Computing RDMs: ')
                for t in range(data_shape[1]):
                    print(t)
                    rdm[t, :] = compute_rdm(data_map[:, t, :, :, :].reshape(data_shape[0], -1))
                    # rdm_rank[t, :, :] = squareform(rankdata(squareform(rdm[t, :, :])))

                tracker['RDMs'][cond] = {}
                tracker['RDMs'][cond]['corr'] = rdm
                # tracker['RDMs'][cond]['rank-corr'] = rdm_rank
                joblib.dump(tracker, tracker_path + tracker_name, compress=True)

                del model_shortened, data_map, rdm


            else:
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

                data_shape = (len(x_train),) + model_shortened.output_shape[1:]
                for att in ['valid', 'invalid']:
                    if att not in tracker['RDMs'][cond]:
                        tracker['RDMs'][cond][att] = {}
                        data_name = data_path + cond + '_' + att + '_' + str(l) + '_data.mmap'

                        if not os.path.isfile(data_name):
                            # Preallocate the memory map

                            data_map = np.memmap(data_name, dtype='float32', shape=data_shape, mode='w+')
                            del data_map

                        if att == 'valid':
                            cue = Ax
                        elif att == 'invalid':
                            cue = Ax_invalid

                        for n in range(len(samples)):
                            print(n)
                            if n == len(samples)-1:
                                idx = np.arange(samples[n], len(x_train))
                            else:
                                idx = np.arange(samples[n],samples[n+1])

                            data_map = np.memmap(data_name, dtype='float32', shape=data_shape, mode='r+')
                            data_map[idx] = predict_generator(model_shortened,
                                                                           [x_train[idx], cue[idx]],
                                                                           y_train[idx],
                                                                           time_steps,
                                                                           batch_size,
                                                                           generator=TimeReplication_onset,
                                                                           steps=len(idx) // batch_size)
                            del data_map

                        data_map = np.memmap(data_name, dtype='float32', shape=data_shape, mode='r+')

                        rdm = np.zeros((data_shape[1], int(data_shape[0] * (data_shape[0]-1) / 2)))

                        print('Computing RDMs: ')
                        for t in range(data_shape[1]):
                            print(t)
                            rdm[t, :] = compute_rdm(data_map[:, t, :, :, :].reshape(data_shape[0], -1))
                            #rdm_rank[t, :, :] = squareform(rankdata(squareform(rdm[t, :, :])))

                        #tracker['RDMs'][cond][att] = {}
                        tracker['RDMs'][cond][att]['corr'] = rdm
                        #tracker['RDMs'][cond]['rank-corr'] = rdm_rank
                        joblib.dump(tracker, tracker_path + tracker_name, compress=True)

                        del data_map, rdm
                del model_shortened

