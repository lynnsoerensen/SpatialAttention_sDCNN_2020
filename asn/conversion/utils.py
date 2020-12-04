import numpy as np
import tensorflow as tf


def transfer_avgPSP(SPre, m_f, h=1, mode=None):
    """ This a transfer function to train a deep convolutional neural network,
    whose activation units can be replaced with spiking units after training
    based on Zambrano et al. 2017.

        S: Signal,potential
        mf: speed of adaptation e.g. [0.03,1]
        h: spike height e.g. 1

    12.05.19: Please note that due to differences in precision between tf and numpy, I recommend to pass tf - tensors
    in the tf mode to get reliable results.
    """
    if mode == 'tf':
        m_f = tf.cast(m_f, dtype=tf.float32)
    Tgamma = 15  # corresponding to adaptation kernel decay of 15 ms.
    # Tphi = 5 #membrane filter in ms
    Teta = 50  # refractory response in ms
    # Tbeta = 50 #PSP in ms
    theta0 = m_f  # Resting potential, e.g. 0.05

    c1 = 2 * m_f * Tgamma ** 2
    c2 = 2 * theta0 * Teta * Tgamma
    c3 = Tgamma * (m_f * Tgamma + 2 * (m_f + 1) * Teta)
    c4 = theta0 * Teta * (Tgamma + Teta)
    if mode == 'tf':
        c0 = h / (tf.exp((c1 * theta0 * 0.5 + c2) / (c3 * theta0 * 0.5 + c4)) - 1)
        out = h / (tf.exp((c1 * SPre + c2) / (c3 * SPre + c4)) - 1) - c0 + h / 2
    else:
        c0 = h / (np.exp((c1 * theta0 * 0.5 + c2) / (c3 * theta0 * 0.5 + c4)) - 1)
        out = h / (np.exp((c1 * SPre + c2) / (c3 * SPre + c4)) - 1) - c0 + h / 2
    return out


def normalize_transfer(m_f, mode=None):  # This assumes that mf is equal to the resting potential
    """ Scales the transfer function so that an input of 1 results in an output of 1
    takes the mf param as input and returns the according h scaling factor"""
    curr = transfer_avgPSP(1, m_f, h=1, mode=mode)
    return 1 / curr


def normalize_weights(weights, BN_weights=None, epsilon=1e-05, h=None):
    """ This function integrates BN parameters with weights and biases based on Rueckauer et al. 2016.
    and/or scale the output spikes of the asn function
    """
    kernel = weights[0]
    bias = weights[1]
    if BN_weights != None:
        # Here we importantly need to convert from variance to SD
        kernel = (BN_weights[0] * kernel) / (np.sqrt(BN_weights[3]) + epsilon)
        bias = (BN_weights[0] / (np.sqrt(BN_weights[3]) + epsilon)) * (bias - BN_weights[2]) + BN_weights[1]
    if h != None:
        kernel = kernel * h

    return [kernel] + [bias]


def predict_model_length(analog_model):
    count = 1 + 4 #Todo: Take 4 away or fix implementation for TimeDist(BN)
    for layer in analog_model.layers:
        #print(layer)
        if layer.__class__.__name__ in ['InputLayer','Flatten','Add']:
            count =count+1
        elif layer.__class__.__name__ == 'ASNTransfer':
            count = count + 2 # 1 for the spiking layer, 1 for the counting
        elif layer.__class__.__name__ == 'Dense':
            config = layer.get_config()
            if config['activation'] in ['softmax','sigmoid']:
                count = count + 1
        elif layer.__class__.__name__ == 'Conv2D':
            if layer._outbound_nodes[0].outbound_layer.name[:3] == 'add':
                count = count + 1
            elif (layer._outbound_nodes[0].outbound_layer.name.startswith('batch_norm')) & \
                    (not layer._outbound_nodes[0].outbound_layer._outbound_nodes[0].outbound_layer.name.startswith('asn')):
                count = count + 1
    return count

