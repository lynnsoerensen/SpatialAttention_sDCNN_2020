import numpy as np
from keras.metrics import top_k_categorical_accuracy
from scipy.signal import gaussian, find_peaks
from scipy.ndimage import filters
from scipy.stats import norm


def top_k_categorical_accuracy_asn(y_pred, y_true, k = 5, num_targets=1):
    """
    Imitates keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5) for numpy arrays
    :param y_pred: prediction by the network
    :param y_true: correct prediction
    :param k: top-k
    :return: accuracy
    Last updated: 20.11.18
    """
    if num_targets == 1:
        target = np.argmax(y_true, axis=-1)
        target = target[:,np.newaxis]
    elif num_targets == 2:
        target = np.argsort(-y_true, axis=-1)[:,:2]
        if np.any(y_true==2):
            idx = np.where(y_true==2)
            target[idx[0], 1] = target[idx[0], 0] # as a missing value
    elif num_targets == 3:
        target = np.argsort(-y_true, axis=-1)[:, :3]
        if np.any(y_true == 2):
            idx = np.where(y_true == 2)
            target[idx[0], 2] = target[idx[0], 0]
        if np.any(y_true==3):
            idx = np.where(y_true == 3)
            target[idx[0], 1:3] = target[idx[0], 0]

    y_pred_ordered = np.argsort(-y_pred, axis=-1)
    out = np.zeros((len(y_pred), num_targets), dtype=bool)
    for i in range(len(y_pred)):
        for j in range(num_targets):
            out[i,j]=np.any(y_pred_ordered[i, :k] == target[i,j])

    return np.mean(out)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def tp_rate(y_predicted, y_true):
    return np.sum(y_predicted[y_true == 1]) / np.sum(y_true == 1)


def fp_rate(y_predicted, y_true):
    return np.sum(y_predicted[y_true == 0]) / np.sum(y_true == 0)

def dPrime(tp_rate, fp_rate):
    # from here: https://lindeloev.net/calculating-d-in-python-and-php/
    Z = norm.ppf
    return Z(tp_rate) - Z(fp_rate)


def obtainTargetLatency(predictionTimecourse, y_true, offset=250):
    """

    :param predictionTimecourse: Prediction time course of output layer
    :param y_true:  Target label.
    :param offset: Time after which to start evaluate. This deals with the initial bias in the network.
    :return: Target detection time, the first moment at which the target prediction was equal or larger than 0.5.
    """
    target_idx = np.where(y_true==1)
    latency = np.zeros(len(y_true))

    for t in range(len(y_true)):
        tps = np.where(predictionTimecourse[target_idx[0][t], offset:, target_idx[1][t]] >= 0.5)[0]
        if len(tps) == 0:
            latency[t] = np.nan
        else:
            latency[t] = np.min(tps) + offset
    return latency


def smooth_response(response, filterWidth=8):
    """

    :param response: Trials x Times
    :param filterWidth: SD of Gaussian
    :return: Smoothed response
    """
    if len(response.shape) == 1:
        response = response[np.newaxis, :]
    gauss = gaussian(10 * filterWidth, filterWidth)
    return filters.convolve1d(response, gauss / gauss.sum(), axis=1)


def estimateLatency(data, data_baseline, start_eval=145, filterWidth=8, show=False, prestimulus_correct=True):
    """

    :param resp: Spike counts, images x times
    :param image_onset: sample when the image started
    :param start_eval: sample from which to start estimating the latency
    :param filterWidth: SD of gaussian for smoothing
    :param baseline_period: Time to estimate the response variability during no stimulation
    :return: Latencies and local maxima per image.
    """
    # Smooth with gaussian of 8 ms
    data = smooth_response(np.array(data, dtype=float), filterWidth=filterWidth)

    # Determine the SEM in the baseline period across all images, This is different from Sundberg et al..
    if prestimulus_correct == True:
        data = data - np.mean(data[:, 50:100], axis=1)[:, np.newaxis]

    latencies = np.full(len(data), np.nan)
    local_maxima = np.full(len(data), np.nan)
    for i in range(len(data)):
        # print(i)
        data_baseline_t = smooth_response(np.array(data_baseline[i].T, dtype=float), filterWidth=filterWidth)
        if prestimulus_correct == True:
            data_baseline_t = data_baseline_t - np.mean(data_baseline_t[:, 50:100], axis=1)[:, np.newaxis]

        # define the criterion
        if show == True:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure()
            sns.distplot(data_baseline_t[:, start_eval:])
            sns.distplot(data[i, start_eval:])
            plt.axvline(
                np.mean(data_baseline_t[:, start_eval:]) + 3.72 * np.std(data_baseline[:, start_eval:]) / np.sqrt(
                    data_baseline_t.shape[0]), color='red')
            plt.axvline(np.percentile(data_baseline_t[:, start_eval:].flatten(), (1 - 0.0001) * 100),
                        color='green')
            plt.show()

        mean_baseline = np.mean(data_baseline_t[:, start_eval:])
        criterion = np.percentile(data_baseline_t[:, start_eval:].flatten(), (1 - 0.0001) * 100)

        # Find all peaks
        l_id, peaks = find_peaks(data[i, start_eval:], height=criterion)

        # Find the mid-point
        if len(l_id) > 0:
            l_id = l_id + start_eval  # bring back to the original time

            midpoint = (peaks['peak_heights'][0] - mean_baseline) * 0.5 + mean_baseline
            # Find zero-crossing
            crossings = np.intersect1d((np.where((data[i, start_eval:l_id[0]] - midpoint) >= 0)[0]),
                                       (np.where((data[i, start_eval:l_id[0]] - midpoint) < 0)[0] + 1))
            if len(crossings) > 0:
                latencies[i] = crossings[-1] + start_eval

            local_maxima[i] = l_id[0]

            # print(str(i) + ': No local minima identified.')

        if show == True:
            plt.figure()
            plt.axhline(criterion,label='criterion', alpha=0.5)
            plt.axvline(start_eval,  alpha=0.5)
            plt.axvline(local_maxima[i], color='red', alpha=0.5)
            plt.axvline(latencies[i], color='pink')

            plt.plot(np.arange(data.shape[1]), data[i])
            plt.plot(np.arange(data.shape[1]), data_baseline_t.T, ls='--', alpha=0.5)
            plt.show()

    return latencies, local_maxima

