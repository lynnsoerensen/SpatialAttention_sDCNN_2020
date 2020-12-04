import numpy as np
from asn.conversion.utils import normalize_transfer
from asn.evaluation.wrappers import predict_generator
from asn.evaluation.generators import TimeReplication
from keras.models import Model
from asn.neuronModel import ASN


def check_spikingModel(spikingModel, analogModel, model_match, inputs, outputs,  n=1000, limit= 0.05,
                       batch_size=2, activations_analog=None, activations=None):
    if isinstance(inputs, list):
        analogInput = inputs[0]
    else:
        analogInput = inputs

    if activations_analog is None:
        activations_analog = obtain_analogActivations(analogModel, analogInput)

    if activations is None:
        activations = obtain_spikingActivations(spikingModel, inputs, outputs, batch_size)

    if spikingModel.layers[1].h_scaling is None:
        h = normalize_transfer(spikingModel.layers[1].mf)
    else:
        h = 1

    all_diffs = []
    offset = 0

    for i, j in enumerate(model_match):
        print('Spiking model layer: ' + str(i+ offset + 1) + ' ' + spikingModel.layers[i+ offset +1].name)
        print('Analog model layer: '+ str(j) + ' ' + analogModel.layers[j].name)
        if spikingModel.layers[i+1].name.startswith('input'):
            offset = offset + 1

        if i == len(model_match)-1:
            diff = np.mean(np.mean(activations[i+offset+1][0, 300:, :], axis=0) - activations_analog[j])
            if abs(diff) < limit:
                print('Last layer: Spiking activation encodes the same values as the analog activation.')
                print('Difference: ' + str(diff))
            else:
                print('Last layer: Spiking and analog activation are not equivalent')
                print('Difference: ' + str((diff)))

            print(np.mean(np.mean(activations[i+offset+1][0, 300:, :], axis=0) - activations_analog[j]))
        else:
            diff = check_spikingLayer(activations[i+offset + 1 ] * h,activations_analog[j], n=n, limit=limit)
        all_diffs.append(diff)
    return activations, activations_analog, all_diffs


def obtain_analogActivations(analogModel, inputs):
    # Obtain the analog activations
    activations_analog = {}
    for l in range(1, len(analogModel.layers)):
        model_shortened = Model(inputs=analogModel.input, outputs=analogModel.layers[l].output)
        activations_analog[l] = model_shortened.predict(inputs)
    return activations_analog


def obtain_spikingActivations(spikingModel, inputs, outputs, batch_size):
    # Obtain the spiking activations
    if isinstance(inputs,list):
        time_steps = spikingModel.input_shape[0][1]
        steps = int(np.ceil(inputs[0].shape[0] / batch_size))
    else:
        time_steps = spikingModel.input_shape[1]
        steps = int(np.ceil(inputs.shape[0] / batch_size))
    activations = {}
    for l in range(len(spikingModel.layers)):
        if spikingModel.layers[l].__class__.__name__ is not 'InputLayer':
            print(l)
            model_shortened = Model(inputs=spikingModel.inputs, outputs=spikingModel.layers[l].output)

            activations[l] = predict_generator(model_shortened, inputs, outputs,
                                               time_steps,
                                               batch_size,
                                               generator=TimeReplication,
                                               steps=steps)
    return activations


def check_spikingLayer(activationSpiking, activationAnalog, bias=[], n=10, limit=0.05):
    diff = []
    for i in range(n):
        # pick a random index
        shape = np.shape(activationAnalog)
        idx = []
        for s in shape:
            idx.append(np.random.randint(s))

        neuron = ASN()
        if len(shape) == 4:
            if len(bias) > 0:
                S_next_neuron, spikes_neuron, S, S_hat, theta = neuron.call(
                    activationSpiking[:, :, np.newaxis, idx[1], idx[2], idx[3]],
                    bias=bias[idx[3]], spike_train=True)
            else:
                S_next_neuron, spikes_neuron, S, S_hat, theta = neuron.call(
                    activationSpiking[:, :, np.newaxis, idx[1], idx[2], idx[3]], spike_train=True)
        elif len(shape) == 2:
            if len(bias) > 0:
                S_next_neuron, spikes_neuron, S, S_hat, theta = neuron.call(activationSpiking[:, :, np.newaxis, idx[1]],
                                                                            bias=bias[idx[3]],
                                                                            spike_train=True)
            else:
                S_next_neuron, spikes_neuron, S, S_hat, theta = neuron.call(activationSpiking[:, :, np.newaxis, idx[1]],
                                                              spike_train=True)


        diff.append(np.mean(S[0, 300:, 0]) - activationAnalog[tuple(idx)])


    if abs(np.mean(np.array(diff))) < limit:
        print('Iteration '+ str(i+1) + ': Spiking activation encodes the same values as the analog activation.')
        print('Difference: ' + str(np.mean(np.array(diff))))
    else:
        print('Iteration '+ str(i+1) +': Spiking and analog activation are not equivalent')
        print('Difference: ' + str(np.mean(np.array(diff))))

    return np.mean(np.array(diff))


