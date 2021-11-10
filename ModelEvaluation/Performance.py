
import joblib
import os
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.models import model_from_json
from keras import optimizers

from asn.utils import load_pickle
from asn.layers.training import ASNTransfer
from asn.conversion.convert import convert_model
from asn.evaluation.wrappers import evaluate_predictions, evaluate_generator
from asn.evaluation.metrics import top_k_categorical_accuracy_asn, top_3_accuracy, top_2_accuracy, tp_rate, fp_rate
from asn.evaluation.generators import coco_squared, TimeReplication_onset
from asn.attention.attn_param import set_attn_param

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.spatial import distance

def load_analogModel(weight_path, model_path):
    # load in the model:
    json_file = open(model_path, 'r')

    loaded_model_json = json_file.read()
    json_file.close()

    modelFT = model_from_json(loaded_model_json, custom_objects={'ASNTransfer': ASNTransfer})
    # load in the weights
    modelFT.load_weights(weight_path)

    modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                    metrics=['binary_accuracy', 'categorical_accuracy', top_2_accuracy, top_3_accuracy])
    return modelFT

dir_path = os.path.abspath('')

# %% Settings
compute_metrics = False
spike_test =False
count_units = False
search_attn_params = False
test_best_attn_params = False
make_split = True
compute_cues = False
mechanism = 'precision' #'inputGain'#'precision'#['inputGain','interactionsPrecision','inputGainPrecision','outputGain','outputGainPrecision','interactions','interactionsPrecision']

np.random.seed(3)

setting = 'FSG_Precision_v0'
dataset = '1_street'

# %% load in the dataset:
img_dir_test = dir_path + '/Datasets/' + dataset + '/images/val2017_single_radial/'

dataset_test = load_pickle(dir_path +'/Datasets/' + dataset + '/val2017_single_radial.pickle')
features = load_pickle(dir_path +
    '/Datasets/' + dataset + '/Features_single_radial.pickle')



targetCentres = [features['targetCentreoM'][s] for s in np.where(features['selection_1'])[0]]

images = [features['x_ids'][s] for s in np.where(features['selection_1'])[0]]
targets = [features['y'][s] for s in np.where(features['selection_1'])[0]]
areas = [features['targetArea'][s] for s in np.where(features['selection_1'])[0]]

# Make sure that only the eligible target will be evaluated by including the selection filter
gen_test = coco_squared(dataset_test, batch_size=np.sum(features['selection_1']), img_dir=img_dir_test,
                        selection=features['selection_1'], shuffle=False)
(x, y) = next(gen_test)

if compute_cues == True:
    # Centers for attention
    Ax = np.zeros((len(x), 2))
    Ax_invalid = np.zeros((len(x), 2))
    areas_sel = np.zeros(len(x))

    train_idx_shuffle = np.arange(len(targetCentres))
    np.random.shuffle(train_idx_shuffle)

    for t in range(len(x)):
        Ax[t, 0] = targetCentres[t][1]
        Ax[t, 1] = targetCentres[t][0]

    # Determine the typical range of object locations, such that extreme locations like 0.01, 0.99 will be avoided.
    Ax_ranges = np.zeros((2,2))
    Ax_ranges[0,:] = np.min(Ax,axis=0)
    Ax_ranges[1,:] = np.max(Ax,axis=0)

    for t in range(len(x)):
        Ax_invalid[t, 0] = targetCentres[train_idx_shuffle[t]][1]
        Ax_invalid[t, 1] = targetCentres[train_idx_shuffle[t]][0]

        while distance.euclidean(Ax[t,:], Ax_invalid[t,:]) < 0.5:
            Ax_invalid[t, 0] = np.random.uniform(Ax_ranges[0,0], Ax_ranges[1,0])
            Ax_invalid[t, 1] = np.random.uniform(Ax_ranges[0,1], Ax_ranges[1, 1])

    cueLocations = {}
    cueLocations['valid'] = Ax
    cueLocations['invalid'] = Ax_invalid
    cueLocations['y'] = y # To check match later.
    joblib.dump(cueLocations, dir_path + '/Datasets/' + dataset + '/cueLocations.pkl')
else:
    cueLocations = joblib.load(dir_path + '/Datasets/' + dataset + '/cueLocations.pkl')

if make_split == True:
    # make a class-balanced random split:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    split.get_n_splits(x, y)

    for train_index, test_index in split.split(x, y):
        print('Split!')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Ax, Ax_test = cueLocations['valid'][train_index], cueLocations['valid'][test_index]
        Ax_invalid, Ax_invalid_test = cueLocations['invalid'][train_index], cueLocations['invalid'][test_index]

else:
    train_index = np.arange(len(x))
    x_train = x
    y_train = y
    Ax = cueLocations['valid']
    Ax_invalid = cueLocations['invalid']


test_images = False
if test_images == True:
    import matplotlib.pyplot as plt

    r = np.round(np.sqrt(np.mean(np.array(areas)) / np.pi) * 224)

    idx = np.random.randint(len(x_train))
    fig, ax = plt.subplots()
    ax.imshow(x_train[idx, :, :, :])
    circle = plt.Circle((Ax[idx, 0] * 224, Ax[idx, 1] * 224), r, alpha=0.2)
    ax.add_artist(circle)
    ax.scatter(Ax[idx, 0] * 224, Ax[idx, 1] * 224, marker='o', c='r')
    ax.scatter((Ax_invalid[idx, 0]) * 224, (Ax_invalid[idx, 1]) * 224, marker='o')
    ax.set_title(y_train[idx])
    fig.savefig(dir_path + '/TESTImage.png')

# IMPORTANT !!!
np.random.seed(3)

# %% clear the memory
del dataset_test, features

# %%  load in the model:
# some model params
optimizer = 'Adam'
optimizer_func = eval('optimizers.' + optimizer)
lr = 1e-4

weight_path = dir_path + '/ModelTraining/' + dataset + '/weights_Adam_constant_0.0001.h5'
model_path = dir_path + '/ModelTraining/' + dataset + "/model.json"
if (compute_metrics == True) | (count_units == True):
    modelFT = load_analogModel(weight_path, model_path)

tracker_name = 'Performance_' + mechanism +  '.pkl'
tracker_path = dir_path + '/ModelEvaluation/'

# Check if there are existing simulations
if os.path.exists(tracker_path + tracker_name):
    tracker = joblib.load(tracker_path + tracker_name)
else:
    tracker = {}
    tracker['num_samples'] = len(x_train)
    tracker['weights'] = weight_path


# %% test it in analog.
if compute_metrics == True:
    scores = modelFT.evaluate(x_train, y_train)
    preds = modelFT.predict(x_train)
    print('Top1-Accuracy analog: ' + str(scores[2]))
    tracker["y_train_single"] = y_train
    tracker['analog_train'] = {}

    tracker['analog_train']['predictions'] = preds
    tracker['analog_train']['binary_acc'] = scores[1]
    tracker['analog_train']['acc'] = scores[2]
    tracker['analog_train']['accTop2'] = scores[3]
    tracker['analog_train']['accTop3'] = scores[4]

    tracker['analog_train']['F1'] = f1_score(y_train, preds >= 0.5,
                                       average='weighted')

    tracker['analog_train']['AUC'] = roc_auc_score(y_train, preds)
    from sklearn.utils import resample
    idx = np.arange(len(x_train))
    auc_scores = np.zeros(500)
    for p in range(500):
        idx_new = resample(idx, n_samples=400,stratify=y_train, random_state=p)

        auc_scores[p] = roc_auc_score(y_train[idx_new], preds[idx_new])

    tracker['analog_train']['tp_rate'] = tp_rate(preds >= 0.5, y_train)
    tracker['analog_train']['fp_rate'] = fp_rate(preds >= 0.5, y_train)

    shuffle_idx = np.arange(y_train.shape[1])
    n_permutations = 10000
    permutation_scores = np.zeros((5, n_permutations))
    for p in range(n_permutations):
        np.random.shuffle(shuffle_idx)
        permutation_scores[0, p] = top_k_categorical_accuracy_asn(preds[:, shuffle_idx], y_train, k=1)
        permutation_scores[1, p] = top_k_categorical_accuracy_asn(preds[:, shuffle_idx], y_train, k=2)
        permutation_scores[2, p] = top_k_categorical_accuracy_asn(preds[:, shuffle_idx], y_train, k=3)
        permutation_scores[3, p] = f1_score(y_train, preds[:, shuffle_idx] >= 0.5, average='weighted')
        permutation_scores[4, p] = roc_auc_score(y_train, preds[:, shuffle_idx])

    tracker['analog_train']['chance_acc'] = np.percentile(permutation_scores[0, :], 95)
    tracker['analog_train']['chance_accTop2'] = np.percentile(permutation_scores[1, :], 95)
    tracker['analog_train']['chance_accTop3'] = np.percentile(permutation_scores[2, :], 95)
    tracker['analog_train']['chance_F1'] = np.percentile(permutation_scores[3, :], 95)
    tracker['analog_train']['chance_AUC'] = np.percentile(permutation_scores[4, :], 95)

    scores = modelFT.evaluate(x_test, y_test)
    preds = modelFT.predict(x_test)
    print('Top1-Accuracy analog: ' + str(scores[2]))
    tracker["y_test_single"] = y_test
    tracker['analog_test'] = {}

    tracker['analog_test']['predictions'] = preds
    tracker['analog_train']['binary_acc'] = scores[1]
    tracker['analog_test']['acc'] = scores[2]
    tracker['analog_test']['accTop2'] = scores[3]
    tracker['analog_test']['accTop3'] = scores[4]

    tracker['analog_test']['F1'] = f1_score(y_test, preds >= 0.5,
                                             average='weighted')

    tracker['analog_test']['AUC'] = roc_auc_score(y_test, preds)
    from sklearn.utils import resample

    idx = np.arange(len(y_test))
    auc_scores = np.zeros(500)
    for p in range(500):
        idx_new = resample(idx, n_samples=400, stratify=y_test, random_state=p)

        auc_scores[p] = roc_auc_score(y_test[idx_new], preds[idx_new])

    tracker['analog_test']['tp_rate'] = tp_rate(preds >= 0.5, y_test)
    tracker['analog_test']['fp_rate'] = fp_rate(preds >= 0.5, y_test)

    shuffle_idx = np.arange(y_train.shape[1])
    n_permutations = 10000
    permutation_scores = np.zeros((5, n_permutations))
    for p in range(n_permutations):
        np.random.shuffle(shuffle_idx)
        permutation_scores[0, p] = top_k_categorical_accuracy_asn(preds[:, shuffle_idx], y_test, k=1)
        permutation_scores[1, p] = top_k_categorical_accuracy_asn(preds[:, shuffle_idx], y_test, k=2)
        permutation_scores[2, p] = top_k_categorical_accuracy_asn(preds[:, shuffle_idx], y_test, k=3)
        permutation_scores[3, p] = f1_score(y_test, preds[:, shuffle_idx] >= 0.5, average='weighted')
        permutation_scores[4, p] = roc_auc_score(y_test, preds[:, shuffle_idx])

    tracker['analog_test']['chance_acc'] = np.percentile(permutation_scores[0, :], 95)
    tracker['analog_test']['chance_accTop2'] = np.percentile(permutation_scores[1, :], 95)
    tracker['analog_test']['chance_accTop3'] = np.percentile(permutation_scores[2, :], 95)
    tracker['analog_test']['chance_F1'] = np.percentile(permutation_scores[3, :], 95)
    tracker['analog_test']['chance_AUC'] = np.percentile(permutation_scores[4, :], 95)

    joblib.dump(tracker, tracker_path + tracker_name,
                compress=True)

# %% test the spiking conversion
mfs = [0.45]#, 0.06]
time_steps = 750
start_eval = 250
batch_size = 2
if spike_test == True:
    tracker['time_steps'] = time_steps
    tracker['start_eval'] = start_eval
    tracker['batch_size'] = batch_size
    #tracker["y_train_single"] = y_train
    empty = set_attn_param()

    for mf in mfs:
        if ("mf_" + str(mf) not in tracker):
            tracker["mf_" + str(mf)] = {}

            modelFT = load_analogModel(weight_path, model_path)

            model_test, model_match = convert_model(modelFT, time_steps, {1: empty}, h_weightscaling=False,
                                                    mf=mf, spike_counting=True)

            model_test.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            print(mf)
            accuracy, predictions = evaluate_generator(model_test, x_train, y_train, time_steps,
                                                       batch_size,
                                                       start_eval=start_eval, eval_mode=1, num_targets=1,
                                                       generator=TimeReplication_onset)

            tracker["mf_" + str(mf)]["predictions"] = predictions[0]
            tracker["mf_" + str(mf)]["spikeCounts"] = predictions[1]

            joblib.dump(tracker, tracker_path + tracker_name,
                        compress=True)

            tracker["mf_" + str(mf)]["acc"] = accuracy
            tracker["mf_" + str(mf)]["accTop2"] = evaluate_predictions(predictions[0],
                                                                       y_train,
                                                                       start_eval,
                                                                       time_steps, 2,
                                                                       num_targets=1)
            tracker["mf_" + str(mf)]["accTop3"] = evaluate_predictions(predictions[0],
                                                                       y_train,
                                                                       start_eval,
                                                                       time_steps, 3,
                                                                       num_targets=1)
            tracker['mf_' + str(mf)]['F1'] = f1_score(y_train, np.mean(
                tracker["mf_" + str(mf)]["predictions"][:, start_eval:time_steps, :],
                axis=1) >= 0.5,
                                                      average='weighted')
            tracker['mf_' + str(mf)]['tp_rate'] = tp_rate(np.mean(
                tracker["mf_" + str(mf)]["predictions"][:, start_eval:time_steps, :],
                axis=1) >= 0.5,
                                                          y_train)

            tracker['mf_' + str(mf)]['fp_rate'] = fp_rate(
                np.mean(
                    tracker["mf_" + str(mf)]["predictions"][:, start_eval:time_steps, :],
                    axis=1) >= 0.5,
                y_train)

            tracker['mf_' + str(mf)]['AUC'] = roc_auc_score(y_train,
                                                            np.mean(tracker["mf_" + str(mf)][
                                                                        "predictions"][:,
                                                                    start_eval:time_steps, :],
                                                                    axis=1))

            joblib.dump(tracker, tracker_path + tracker_name,
                        compress=True)
            K.clear_session()


if count_units == True:

    mf = 0.18

    empty = set_attn_param()
    model_test, model_match = convert_model(modelFT, time_steps, {1: empty}, h_weightscaling=False,
                                             mf=mf, spike_counting=False)

    dimensions = []
    for layer in model_test.layers:
        if layer.name.startswith('asn'):
            dimensions.append(layer.output_shape)

    tracker['dimensions'] = dimensions

    joblib.dump(tracker, tracker_path + tracker_name,
                compress=True)

    unitCount = 0
    spikeCount = 0
    for i, d in enumerate(tracker['dimensions']):
        if len(d) == 5:
            unitCount = unitCount + d[2] * d[3] * d[4]
        elif len(d) == 3:
            unitCount = unitCount + d[1]
        spikeCount = spikeCount + np.mean(tracker['mf_0.45']['spikeCounts'][:, i])

    modelFR = (spikeCount/unitCount) * 1000/time_steps

# %%  convert it to spikes with different attentional biases
if search_attn_params == True:
    # The constants
    mf = 0.45
    mode = 'multiLayer'
    b_idx = 8
    blocks = np.arange(1, 9)
    block_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    layer_idx = np.array([3, 7, 11, 16, 20, 25, 29, 34])
    stride_sizes = np.array([4, 4, 8, 8, 16, 16, 32, 32])
    AxWidths = np.array([40])

    if 'attention_' + str(mf) not in tracker:
        tracker['attention_' + str(mf)] = {}
        tracker['attention_' + str(mf)]['mode']= mode
        tracker['attention_' + str(mf)]['b_idx'] = b_idx
        tracker['attention_' + str(mf)]['layers'] = layer_idx[:b_idx]
        tracker['attention_' + str(mf)]['AxWidth'] = AxWidths[0]
    # The changes
    if mechanism == 'inputGain':
        precisions = [0]
        gains_input = [0, 0.05,  0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        gains_output = [0]
    elif mechanism == 'outputGain':
        precisions = [0]
        gains_output = [0, 0.05,  0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        gains_input = [0]
    elif mechanism == 'precision':
        precisions = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        gains_input = [0]
        gains_output = [0]
    else:
        raise ValueError('This mechanism is not specified.')

    for gain_output in gains_output:
        for gain_input in gains_input:
            for precision in precisions:

                cond = 'P-' + str(precision) + '_I-' + str(gain_input) + '_O-' + str(gain_output)
                if cond not in tracker['attention_' + str(mf)]:
                    tracker['attention_' + str(mf)][cond] = {}

                if mode == 'multiLayerReverse':
                    successive = {}
                    for b in range(len(blocks), 0,
                                   -1):  # Note that we are not evaluating performance when everything is changed anymore, this is identical to multiLayer
                        print(b)
                        successive[b] = layer_idx[block_idx >= b]

                elif mode == 'multiLayerReverse-1':
                    successive = {}
                    top = 5
                    for b in range(len(blocks) - 1, 0,
                                   -1):  # Note that we are not evaluating performance when everything is changed anymore, this is identical to multiLayer
                        successive[b] = layer_idx[(block_idx >= b) & (block_idx <= top)]

                elif mode == 'multiLayer-1':
                    successive = {}
                    bottom = 2
                    for b in range(bottom, len(
                            blocks) + 1):  # Note that we are not evaluating performance when everything is changed anymore, this is identical to multiLayer
                        successive[b] = layer_idx[(block_idx <= b) & (block_idx >= bottom)]
                else:
                    successive = {}
                    for b in range(1, len(blocks) + 1):
                        successive[b] = layer_idx[block_idx <= b]

                a_idx = 0
                AxWidth = AxWidths[a_idx]

                if mode in ['multiLayer', 'multiLayerReverse', 'multiLayerReverse-1', 'multiLayer-1']:
                    l = successive[b_idx]
                    stride_current = {}
                    AxWidths_current = {}
                    IxWidths_current = {}
                    for layer in range(len(l)):
                        stride_current[layer] = stride_sizes[layer_idx == l[layer]]

                        AxWidths_current[layer] = np.array(np.ceil(AxWidths / stride_current[layer]), dtype=int)

                else:
                    l = layer_idx[block_idx == b_idx]
                    stride_current = {}
                    AxWidths_current = {}
                    IxWidths_current = {}
                    for layer in range(len(l)):
                        stride_current[layer] = stride_sizes[layer_idx == l[layer]]
                        AxWidths_current[layer] = np.array(np.ceil(AxWidths / stride_current[layer]), dtype=int)

                if ('valid' not in tracker['attention_' + str(mf)][cond]): #| ('invalid' not in tracker['attention_' + str(mf)][cond]):

                    np.random.seed(3)
                    tf.set_random_seed(3)

                    # %% Load the base-model
                    modelFT = load_analogModel(weight_path, model_path)

                    if mode in ['multiLayer', 'multiLayerReverse', 'multiLayerReverse-1', 'singleLayer',
                                'multiLayer-1']:
                        attn_param = {}
                        for layer in range(len(l)):
                            attn_param[layer] = set_attn_param(layer_idx=l[layer],
                                                               AxWidth1=AxWidths_current[layer][0],
                                                               AxWidth2=AxWidths_current[layer][0],
                                                               InputGain=gain_input, OutputGain=gain_output,
                                                               Precision=precision)
                    else:
                        attn_param = set_attn_param(layer_idx=l, AxWidth1=AxWidths_current[0],
                                                    AxWidth2=AxWidths_current[0], InputGain=gain_input,
                                                    OutputGain=gain_output,
                                                    Precision=precision)
                        attn_param = {l: attn_param}

                    model_test, model_match = convert_model(modelFT, time_steps, attn_param, h_weightscaling=False,
                                                            mf=mf, spike_counting=True)

                    model_test.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                                       metrics=['accuracy'])


                    att_cond = 'valid'

                    print('Evaluating attentional benefits for condition ' + cond + '-' + att_cond)

                    np.random.seed(3)


                    if att_cond == 'valid':

                        accuracy, predictions = evaluate_generator(model_test, [x_train, Ax],
                                                                   y_train,
                                                                   time_steps, batch_size,
                                                                   start_eval=start_eval,
                                                                   eval_mode=1, num_targets=1,
                                                                   generator=TimeReplication_onset)
                    elif att_cond == 'invalid':
                        accuracy, predictions = evaluate_generator(model_test,
                                                                   [x_train, Ax_invalid], y_train,
                                                                   time_steps, batch_size,
                                                                   start_eval=start_eval,
                                                                   eval_mode=1, num_targets=1,
                                                                   generator=TimeReplication_onset)

                    tracker['attention_' + str(mf)][cond][att_cond] = {}

                    tracker['attention_' + str(mf)][cond][att_cond]["predictions"] = predictions[0]
                    tracker['attention_' + str(mf)][cond][att_cond]["spikeCounts"] = predictions[1]

                    joblib.dump(tracker, tracker_path + tracker_name,
                                compress=True)

                    tracker['attention_' + str(mf)][cond][att_cond]["acc"] = accuracy
                    tracker['attention_' + str(mf)][cond][att_cond]["accTop2"] = evaluate_predictions(
                        predictions[0],
                        y_train,
                        start_eval,
                        time_steps, 2,
                        num_targets=1)
                    tracker['attention_' + str(mf)][cond][att_cond]["accTop3"] = evaluate_predictions(
                        predictions[0],
                        y_train,
                        start_eval,
                        time_steps, 3,
                        num_targets=1)

                    tracker['attention_' + str(mf)][cond][att_cond]['F1'] = f1_score(y_train, np.mean(
                        predictions[0][:, start_eval:time_steps, :],
                        axis=1) >= 0.5, average='weighted')

                    tracker['attention_' + str(mf)][cond][att_cond]['tp_rate'] = tp_rate(
                        np.mean(predictions[0][:, start_eval:time_steps, :],
                                axis=1) >= 0.5, y_train)

                    tracker['attention_' + str(mf)][cond][att_cond]['fp_rate'] = fp_rate(
                        np.mean(predictions[0][:, start_eval:time_steps,
                                :], axis=1) >= 0.5, y_train)

                    tracker['attention_' + str(mf)][cond][att_cond]['AUC'] = roc_auc_score(y_train,
                                                                                           np.mean(predictions[0][:,
                                                                                                   start_eval:time_steps,
                                                                                                   :], axis=1))

                    joblib.dump(tracker, tracker_path + tracker_name,
                                compress=True)

                    K.clear_session()


if test_best_attn_params == True:
    # Identify the best performing model
    # The constants
    mf = 0.45
    mode = 'multiLayer'
    b_idx = 8
    blocks = np.arange(1, 9)
    block_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    layer_idx = np.array([3, 7, 11, 16, 20, 25, 29, 34])
    stride_sizes = np.array([4, 4, 8, 8, 16, 16, 32, 32])
    AxWidths = np.array([40])
    start_eval = 250
    time_steps = 750

    conds = [key for key in tracker['attention_' + str(mf)].keys() if key.startswith('P-') ]
    condBest = None
    bestAUC = 0
    # Count through keys here
    for cond in conds:
        if tracker['attention_' + str(mf)][cond]['valid']['AUC'] > bestAUC:
            condBest = cond
            bestAUC = tracker['attention_' + str(mf)][cond]['valid']['AUC']


    tracker['attention_' + str(mf)]['testSet'] = {}
    tracker['attention_' + str(mf)]['testSet']['bestParams'] = condBest

    # Evaluate model performance for the best training parameters.
    strs = condBest.split('_')
    for str_i in strs:
        if str_i.startswith('P-'):
            precision = float(str_i[2:])
        elif str_i.startswith('I-'):
            gain_input = float(str_i[2:])

        elif str_i.startswith('O-'):
            gain_output = float(str_i[2:])
        else:
            raise ValueError()

    if mode == 'multiLayerReverse':
        successive = {}
        for b in range(len(blocks), 0,
                       -1):  # Note that we are not evaluating performance when everything is changed anymore, this is identical to multiLayer
            print(b)
            successive[b] = layer_idx[block_idx >= b]

    elif mode == 'multiLayerReverse-1':
        successive = {}
        top = 5
        for b in range(len(blocks) - 1, 0,
                       -1):  # Note that we are not evaluating performance when everything is changed anymore, this is identical to multiLayer
            successive[b] = layer_idx[(block_idx >= b) & (block_idx <= top)]

    elif mode == 'multiLayer-1':
        successive = {}
        bottom = 2
        for b in range(bottom, len(
                blocks) + 1):  # Note that we are not evaluating performance when everything is changed anymore, this is identical to multiLayer
            successive[b] = layer_idx[(block_idx <= b) & (block_idx >= bottom)]
    else:
        successive = {}
        for b in range(1, len(blocks) + 1):
            successive[b] = layer_idx[block_idx <= b]

    a_idx = 0
    AxWidth = AxWidths[a_idx]

    if mode in ['multiLayer', 'multiLayerReverse', 'multiLayerReverse-1', 'multiLayer-1']:
        l = successive[b_idx]
        stride_current = {}
        AxWidths_current = {}
        IxWidths_current = {}
        for layer in range(len(l)):
            stride_current[layer] = stride_sizes[layer_idx == l[layer]]

            AxWidths_current[layer] = np.array(np.ceil(AxWidths / stride_current[layer]), dtype=int)

    else:
        l = layer_idx[block_idx == b_idx]
        stride_current = {}
        AxWidths_current = {}
        IxWidths_current = {}
        for layer in range(len(l)):
            stride_current[layer] = stride_sizes[layer_idx == l[layer]]
            AxWidths_current[layer] = np.array(np.ceil(AxWidths / stride_current[layer]), dtype=int)

    if ('valid' not in tracker['attention_' + str(mf)]['testSet']) | (
            'invalid' not in tracker['attention_' + str(mf)]['testSet']):

        np.random.seed(3)
        tf.set_random_seed(3)

        # %% Load the base-model
        modelFT = load_analogModel(weight_path, model_path)

        if mode in ['multiLayer', 'multiLayerReverse', 'multiLayerReverse-1', 'singleLayer',
                    'multiLayer-1']:
            attn_param = {}
            for layer in range(len(l)):
                attn_param[layer] = set_attn_param(layer_idx=l[layer],
                                                   AxWidth1=AxWidths_current[layer][0],
                                                   AxWidth2=AxWidths_current[layer][0],
                                                   InputGain=gain_input, OutputGain=gain_output,
                                                   Precision=precision)
        else:
            attn_param = set_attn_param(layer_idx=l, AxWidth1=AxWidths_current[0],
                                        AxWidth2=AxWidths_current[0], InputGain=gain_input,
                                        OutputGain=gain_output,
                                        Precision=precision)
            attn_param = {l: attn_param}

        model_test, model_match = convert_model(modelFT, time_steps, attn_param, h_weightscaling=False,
                                                mf=mf, spike_counting=True)

        model_test.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                           metrics=['accuracy'])

        for att_cond in ['valid', 'invalid']:

            if att_cond not in tracker['attention_' + str(mf)]['testSet']:
                print('Evaluating attentional benefits for condition ' + condBest + '-' + att_cond)

                np.random.seed(3)

                if att_cond == 'valid':

                    accuracy, predictions = evaluate_generator(model_test, [x_test, Ax_test],
                                                               y_test,
                                                               time_steps, batch_size,
                                                               start_eval=start_eval,
                                                               eval_mode=1, num_targets=1,
                                                               generator=TimeReplication_onset)
                elif att_cond == 'invalid':
                    accuracy, predictions = evaluate_generator(model_test,
                                                               [x_test, Ax_invalid_test], y_test,
                                                               time_steps, batch_size,
                                                               start_eval=start_eval,
                                                               eval_mode=1, num_targets=1,
                                                               generator=TimeReplication_onset)

                tracker['attention_' + str(mf)]['testSet'][att_cond] = {}

                tracker['attention_' + str(mf)]['testSet'][att_cond]["predictions"] = predictions[0]
                tracker['attention_' + str(mf)]['testSet'][att_cond]["spikeCounts"] = predictions[1]

                joblib.dump(tracker, tracker_path + tracker_name,
                            compress=True)

                tracker['attention_' + str(mf)]['testSet'][att_cond]["acc"] = accuracy
                tracker['attention_' + str(mf)]['testSet'][att_cond]["accTop2"] = evaluate_predictions(
                    predictions[0],
                    y_test,
                    start_eval,
                    time_steps, 2,
                    num_targets=1)
                tracker['attention_' + str(mf)]['testSet'][att_cond]["accTop3"] = evaluate_predictions(
                    predictions[0],
                    y_test,
                    start_eval,
                    time_steps, 3,
                    num_targets=1)

                tracker['attention_' + str(mf)]['testSet'][att_cond]['F1'] = f1_score(y_test, np.mean(
                    predictions[0][:, start_eval:time_steps, :],
                    axis=1) >= 0.5, average='weighted')

                tracker['attention_' + str(mf)]['testSet'][att_cond]['tp_rate'] = tp_rate(
                    np.mean(predictions[0][:, start_eval:time_steps, :],
                            axis=1) >= 0.5, y_test)

                tracker['attention_' + str(mf)]['testSet'][att_cond]['fp_rate'] = fp_rate(
                    np.mean(predictions[0][:, start_eval:time_steps,
                            :], axis=1) >= 0.5, y_test)

                tracker['attention_' + str(mf)]['testSet'][att_cond]['AUC'] = roc_auc_score(y_test,
                                                                                            np.mean(predictions[0][:,
                                                                                                    start_eval:time_steps,
                                                                                                    :], axis=1))

                joblib.dump(tracker, tracker_path + tracker_name,
                            compress=True)

        K.clear_session()
