from __future__ import print_function
import numpy as np
from keras.layers import Input, Flatten,TimeDistributed
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.merge import Add, Subtract, Concatenate
from keras.models import Model
from asn.layers.test import *
from asn.conversion.utils import normalize_weights
from asn.attention.attn_param import set_model_attn_param


def convert_architecture(model_training, time_steps, mf_base, h_scaling, attn_param, skip=[], skip_value=0.06, spike_counting=True):
    """model_training: analog Keras model
    time_steps: time_steps for digital/spiking conversion
    mf: mf: precision for spike generation. Default: 0.1
    h_scaling: mode for spike generation. Are spikes scales by h?
    resnet: is the model_training a resnet?
    attn_param: dict with parameters for attention modulation

    model_test: spiking architecture
    Last updated: 23.10.18
    """

    print('Translating the following architecture to ASN:')
    model_training.summary()

    input_shape = model_training.get_input_shape_at(0)  # Input size of training model

    if len(input_shape) != 4:
        raise Exception("Input shape of analog model should be a tuple (nb_rows, nb_cols, nb_channels)")
    input_shape = (time_steps,) + input_shape[1:4]  # Add time dimension

    input = Input(shape=input_shape)
    input_att = Input(shape=((2,)))  # for the parameter Ax1 & Ax2
    x = input
    if model_training.layers[0].__class__.__name__ == 'InputLayer':
        l_train = 1
    else:
        l_train = 0
    model_match = []
    l_test = 1  # Since the first layer is the input layer
    attention_applied = False
    for i in np.arange(0, len(model_training.layers)):
        mode = model_training.layers[i].__class__.__name__
        print('Evaluating layer ' + str(i) + ' - ' + mode)

        if i < l_train:
            print('Information from layer ' + str(i) + ' (' + mode + ') has already been integrated')
        else:

            if attn_param[l_test]["layer_idx"] is not None:
                print('Applying attention to layer ' + str(l_test))

                x = [x, input_att]
                print(str(x[0].shape))
                attention_applied = True
                ASN_2D_layer = ASN_2D_attention

            else:
                ASN_2D_layer = ASN_2D
                print(str(x.shape))

            if l_test == 1:
                input_layer = True  # layer will integrate current instead of spikes
            else:
                input_layer = False

            if l_test in skip:
                mf = skip_value
            else:
                mf = mf_base

            if mode == 'BatchNormalization':
                if model_training.layers[l_train + 1].__class__.__name__ == 'ASNTransfer':
                    if model_training.layers[l_train + 2].__class__.__name__ == 'MaxPooling2D':
                        print('Building joint BN-ASN-MaxPool Layer as layer ' + str(
                            l_test) + ' in model_test')

                        if attn_param[l_test]["layer_idx"] is not None:
                            n_filter = x[0].shape[-1].value
                        else:
                            n_filter = x.shape[-1].value

                        pool_size = model_training.layers[l_train + 2].pool_size
                        pool_stride = model_training.layers[l_train + 2].strides
                        pool_padding = model_training.layers[l_train + 2].padding
                        x = ASN_2D_layer(filters=n_filter,
                                           padding=model_training.layers[l_train].padding, use_bias=True,
                                           pool_mode='max', pool_size=pool_size, pool_stride=pool_stride,
                                           pool_padding=pool_padding, kernel_initializer='ones',
                                           bias_initializer='zeros', mf=mf, h_scaling=h_scaling, input_layer=input_layer,
                                           attn_param=attn_param[l_test])(x)
                        if spike_counting == True:
                            # Add it to the spike count
                            if l_test == 1:
                                spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                            else:
                                s = SpikeSum(h_scaling=h_scaling)(x)
                                spike_counter = Concatenate()([spike_counter, s])

                        l_train = l_train + 3  # because now 4 layers were combined
                        l_test = l_test + 1
                        model_match.append(l_train)

                    elif model_training.layers[l_train + 2].__class__.__name__ == 'AveragePooling2D':
                        print('Building joint BN-ASN-AvgPool Layer as layer ' + str(
                            l_test) + ' in model_test')

                        if attn_param[l_test]["layer_idx"] is not None:
                            n_filter = x[0].shape[-1].value
                        else:
                            n_filter = x.shape[-1].value

                        pool_size = model_training.layers[l_train + 2].pool_size
                        pool_stride = model_training.layers[l_train + 2].strides
                        pool_padding = model_training.layers[l_train + 2].padding

                        x = ASN_2D_layer(filters=n_filter, use_bias=True,
                                         pool_mode='avg', pool_size=pool_size, pool_stride=pool_stride,
                                         kernel_initializer='ones',
                                         bias_initializer='zeros',
                                         pool_padding=pool_padding, mf=mf, h_scaling=h_scaling,
                                         input_layer=input_layer, attn_param=attn_param[l_test])(x)

                        if spike_counting == True:
                            if l_test == 1:
                                spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                            else:
                                s = SpikeSum(h_scaling=h_scaling)(x)
                                spike_counter = Concatenate()([spike_counter, s])

                        l_train = l_train + 3  # because now 3 layers were combined
                        l_test = l_test + 1

                        model_match.append(l_train)

                    else:
                        print('Building joint BN-ASN Layer as layer ' + str(l_test) + ' in model_test')
                        # Diagnose how high-dimensional the input is:
                        if attn_param[l_test]["layer_idx"] is not None:
                            dimensionality = x[0].shape.ndims
                            n_filter = x[0].shape[-1].value
                        else:
                            dimensionality = x.shape.ndims
                            n_filter = x.shape[-1].value

                        if dimensionality > 3:
                            x = ASN_2D_layer(filters=n_filter, use_bias=True, mf=mf, kernel_initializer='ones',
                                               bias_initializer='zeros', input_layer=input_layer, h_scaling=h_scaling,
                                               attn_param=attn_param[l_test])(x)
                        else:
                            x = ASN_1D(units=n_filter, mf=mf, use_bias=True, kernel_initializer='ones',
                                       bias_initializer='zeros', input_layer=input_layer, h_scaling=h_scaling)(x)
                        if spike_counting == True:
                            if l_test == 1:
                                spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                            else:
                                s = SpikeSum(h_scaling=h_scaling)(x)
                                spike_counter = Concatenate()([spike_counter, s])

                        # Do the accounting.
                        l_train = l_train + 2  # because 2 layers were added
                        l_test = l_test + 1
                        model_match.append(l_train)

                else:
                    print('BN Layer will be integrated with other layers')
                    l_train = l_train + 1
                    l_test = l_test

            elif mode == 'ASNTransfer':
                print('Building ASN Layer as layer ' + str(l_test) + ' in model_test')
                # Diagnose how high-dimensional the input is:
                if attn_param[l_test]["layer_idx"] is not None:
                    dimensionality = x[0].shape.ndims
                    n_filter = x[0].shape[-1].value
                else:
                    dimensionality = x.shape.ndims
                    n_filter = x.shape[-1].value

                if dimensionality > 3:
                    x = ASN_2D_layer(filters=n_filter, use_bias=True, mf=mf, kernel_initializer='ones',
                                     bias_initializer='zeros', input_layer=input_layer, h_scaling=h_scaling,
                                     attn_param=attn_param[l_test])(x)
                else:
                    x = ASN_1D(units=n_filter, mf=mf, use_bias=True, kernel_initializer='ones',
                               bias_initializer='zeros', input_layer=input_layer, h_scaling=h_scaling)(x)
                if spike_counting == True:
                    if l_test == 1:
                        spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                    else:
                        s = SpikeSum(h_scaling=h_scaling)(x)
                        spike_counter = Concatenate()([spike_counter, s])

                l_train = l_train + 1
                l_test = l_test + 1
                model_match.append(l_train)

            elif mode == 'Conv2D':
                if (model_training.layers[l_train + 1].__class__.__name__ == 'BatchNormalization') & \
                        (model_training.layers[l_train]._outbound_nodes[0].outbound_layer.name.startswith(
                        'batch_norm')): # This second part tests the connection.
                    if model_training.layers[l_train + 2].__class__.__name__ == 'ASNTransfer':
                        if model_training.layers[l_train + 3].__class__.__name__ == 'MaxPooling2D':
                            print('Building joint Conv2D-BN-ASN-MaxPool Layer as layer ' + str(
                                l_test) + ' in model_test')
                            pool_size = model_training.layers[l_train + 3].pool_size
                            pool_stride = model_training.layers[l_train + 3].strides
                            pool_padding = model_training.layers[l_train + 3].padding
                            x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                                       filters=model_training.layers[l_train].filters,
                                       strides=model_training.layers[l_train].strides,
                                       padding=model_training.layers[l_train].padding, use_bias=True,
                                       pool_mode='max', pool_size=pool_size, pool_stride=pool_stride,
                                       pool_padding=pool_padding,
                                       mf=mf, h_scaling=h_scaling, input_layer=input_layer,
                                       attn_param=attn_param[l_test])(x)

                            if spike_counting == True:
                                if l_test == 1:
                                    spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                                else:

                                    s = SpikeSum(h_scaling=h_scaling)(x)
                                    spike_counter = Concatenate()([spike_counter, s])

                            l_train = l_train + 4  # because now 4 layers were combined
                            l_test = l_test + 1
                            model_match.append(l_train)

                        elif model_training.layers[l_train + 3].__class__.__name__ == 'AveragePooling2D':
                            print('Building joint Conv2D-BN-ASN-AvgPool Layer as layer ' + str(
                                l_test) + ' in model_test')
                            pool_size = model_training.layers[l_train + 3].pool_size
                            pool_stride = model_training.layers[l_train + 3].strides
                            pool_padding = model_training.layers[l_train + 3].padding
                            x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                                       filters=model_training.layers[l_train].filters,
                                       strides=model_training.layers[l_train].strides,
                                       padding=model_training.layers[l_train].padding, use_bias=True,
                                       pool_mode='avg', pool_size=pool_size, pool_stride=pool_stride,
                                       pool_padding=pool_padding, mf=mf, h_scaling=h_scaling,
                                       input_layer=input_layer, attn_param=attn_param[l_test])(x)
                            if spike_counting == True:
                                if l_test == 1:
                                    spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                                else:
                                    s = SpikeSum(h_scaling=h_scaling)(x)
                                    spike_counter = Concatenate()([spike_counter, s])

                            l_train = l_train + 4  # because now 4 layers were combined
                            l_test = l_test +  1
                            model_match.append(l_train)

                        elif model_training.layers[l_train + 3].__class__.__name__ in ['GaussianDropout', 'Dropout', 'GaussianNoise','UniformNoise']:
                            if model_training.layers[l_train + 4].__class__.__name__ == 'MaxPooling2D':
                                print(
                                    'Skipping DropoutLayer and building joint Conv2D-BN-ASN-MaxPool Layer as layer ' + str(
                                        l_test) + ' in model_test')
                                pool_size = model_training.layers[l_train + 4].pool_size
                                pool_stride = model_training.layers[l_train + 4].strides
                                pool_padding = model_training.layers[l_train + 3].padding
                                x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                                           filters=model_training.layers[l_train].filters,
                                           strides=model_training.layers[l_train].strides,
                                           padding=model_training.layers[l_train].padding, use_bias=True,
                                           pool_mode='max', pool_size=pool_size, pool_stride=pool_stride,
                                           pool_padding=pool_padding, mf=mf, h_scaling=h_scaling,
                                           input_layer=input_layer, attn_param=attn_param[l_test])(x)
                                if spike_counting == True:
                                    if l_test == 1:
                                        spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                                    else:
                                        s = SpikeSum(h_scaling=h_scaling)(x)
                                        spike_counter = Concatenate()([spike_counter, s])

                                l_train = l_train + 5  # because now 5 layers were combined
                                l_test = l_test + 1

                                model_match.append(l_train)
                            elif model_training.layers[l_train + 4].__class__.__name__ == 'AveragePooling2D':
                                print(
                                    'Skipping DropoutLayer and building joint Conv2D-BN-ASN-AvgPool Layer as layer ' + str(
                                        l_test) + ' in model_test')
                                pool_size = model_training.layers[l_train + 4].pool_size
                                pool_stride = model_training.layers[l_train + 4].strides
                                pool_padding = model_training.layers[l_train + 3].padding
                                x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                                           filters=model_training.layers[l_train].filters,
                                           strides=model_training.layers[l_train].strides,
                                           padding=model_training.layers[l_train].padding, use_bias=True,
                                           pool_mode='avg', pool_size=pool_size, pool_stride=pool_stride,
                                           pool_padding=pool_padding,
                                           mf=mf, h_scaling=h_scaling, input_layer=input_layer,
                                           attn_param=attn_param[l_test])(x)

                                if spike_counting == True:
                                    if l_test == 1:
                                        spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                                    else:
                                        s = SpikeSum(h_scaling=h_scaling)(x)
                                        spike_counter = Concatenate()([spike_counter, s])

                                l_train = l_train + 5  # because now 5 layers were combined
                                l_test = l_test + 1
                                model_match.append(l_train)

                            else:
                                print(
                                    'Building joint Conv2D-BN-ASN Layer as layer ' + str(l_test) + ' in model_test')
                                x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                                           padding=model_training.layers[l_train].padding,
                                           filters=model_training.layers[l_train].filters,
                                           use_bias=True, mf=mf, h_scaling=h_scaling,
                                           input_layer=input_layer, attn_param=attn_param[l_test])(x)
                                if spike_counting == True:
                                    if l_test == 1:
                                        spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                                    else:
                                        s = SpikeSum(h_scaling=h_scaling)(x)
                                        spike_counter = Concatenate()([spike_counter, s])

                                l_train = l_train + 4  # because now 4 layers were combined
                                l_test = l_test + 1
                                model_match.append(l_train)

                        else:
                            print('Building joint Conv2D-BN-ASN Layer as layer ' + str(l_test) + ' in model_test')
                            x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                                       padding=model_training.layers[l_train].padding,
                                       filters=model_training.layers[l_train].filters,
                                       strides=model_training.layers[l_train].strides,
                                       use_bias=True, mf=mf, h_scaling=h_scaling,
                                       input_layer=input_layer, attn_param=attn_param[l_test])(x)

                            if spike_counting == True:
                                if l_test == 1:
                                    spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                                else:
                                    s = SpikeSum(h_scaling=h_scaling)(x)
                                    spike_counter = Concatenate()([spike_counter, s])


                            l_train = l_train + 3  # because now 3 layers were combined
                            l_test = l_test + 1
                            model_match.append(l_train)

                    else:
                        print('Building joint Conv2D-BN Layer as layer ' + str(l_test) + ' in model_test')
                        x = TimeDistributed(Conv2D(model_training.layers[l_train].filters,
                                                   model_training.layers[l_train].kernel_size,
                                                   strides=model_training.layers[l_train].strides,
                                                   padding=model_training.layers[l_train].padding,
                                                   use_bias= False))(x)

                        l_train = l_train + 2
                        l_test = l_test + 1
                        model_match.append(l_train)

                elif model_training.layers[l_train + 1].__class__.__name__ == 'ASNTransfer':
                    print('Building joint Conv2D-ASN Layer as layer ' + str(l_test) + ' in model_test')

                    x = ASN_2D_layer(kernel_size=model_training.layers[l_train].kernel_size,
                               filters=model_training.layers[l_train].filters,
                               strides=model_training.layers[l_train].strides,
                               use_bias=True, mf=mf, h_scaling=h_scaling, input_layer=input_layer,
                               attn_param=attn_param[l_test])(x)
                    if spike_counting == True:
                        if l_test == 1:
                            spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                        else:
                            s = SpikeSum(h_scaling=h_scaling)(x)
                            spike_counter = Concatenate()([spike_counter, s])


                    l_train = l_train + 2
                    l_test = l_test + 1
                    model_match.append(l_train)

                elif model_training.layers[l_train + 1].__class__.__name__ in ['Conv2D', 'Add', 'BatchNormalization']:

                    if model_training.layers[l_train].input.name.startswith(identity_name):  # Identity
                        print('Building time-distributed Conv2D as layer ' + str(l_test) + ' in model_test for identity')
                        identity = TimeDistributed(Conv2D(model_training.layers[l_train].filters,
                                                          model_training.layers[l_train].kernel_size,
                                                          padding=model_training.layers[l_train].padding,
                                                          strides=model_training.layers[l_train].strides,
                                                          use_bias=False))(identity)

                        # Update the identity name
                        identity_name = model_training.layers[i].name

                    else:
                        print('Building time-distributed Conv2D as layer ' + str(l_test) + ' in model_test')  # Main
                        x = TimeDistributed(Conv2D(model_training.layers[l_train].filters,
                                                   model_training.layers[l_train].kernel_size,
                                                   strides=model_training.layers[l_train].strides,
                                                   padding=model_training.layers[l_train].padding,
                                                   use_bias=False))(x)

                    l_train = l_train + 1
                    l_test = l_test + 1
                    if model_training.layers[l_train - 1]._outbound_nodes[0].outbound_layer.name.startswith(
                            'batch_normalization'):
                        from asn.utils import getLayerIndexByName
                        model_match.append(getLayerIndexByName(model_training,
                                                               model_training.layers[l_train - 1]._outbound_nodes[
                                                                   0].outbound_layer.name) + 1)
                    else:
                        model_match.append(l_train)
                else:
                    print('Building time-distributed Conv2D as layer ' + str(l_test) + ' in model_test')  # Main
                    x = TimeDistributed(Conv2D(model_training.layers[l_train].filters,
                                               model_training.layers[l_train].kernel_size,
                                               strides=model_training.layers[l_train].strides,
                                               padding=model_training.layers[l_train].padding,
                                               use_bias=False))(x)
                    l_train = l_train + 1
                    l_test = l_test + 1

                    if model_training.layers[l_train]._outbound_nodes[0].outbound_layer.name.startswith(
                            'batch_normalization'):
                        from asn.utils import getLayerIndexByName
                        model_match.append(getLayerIndexByName(model_training,
                                                               model_training.layers[l_train]._outbound_nodes[
                                                                   0].outbound_layer.name))
                    else:
                        model_match.append(l_train)

            elif mode == 'MaxPooling2D':
                print('Building time-distributed ' + mode + ' as layer ' + str(l_test) + ' in model_test')
                pool_size = model_training.layers[l_train].pool_size
                x = TimeDistributed(MaxPooling2D(pool_size=pool_size))(x)

                l_train = l_train + 1
                l_test = l_test + 1
                model_match.append(l_train)

            elif mode == 'Add':
                print('Building ' + mode + ' as layer ' + str(l_test) + ' in model_test')
                x = Add()([x, identity])
                l_train = l_train + 1
                l_test = l_test + 1
                model_match.append(l_train)

            elif mode == 'Subtract':
                print('Building ' + mode + ' as layer ' + str(l_test) + ' in model_test')
                x = Subtract()([identity, x])
                l_train = l_train + 1
                l_test = l_test + 1
                model_match.append(l_train)

            elif mode == 'Dense':

                if len(model_training.layers) == l_train + 1:  # Make a softmax output layer for the last layer
                    nodes = model_training.layers[l_train].output_shape[1]
                    config = model_training.layers[l_train].get_config()
                    activation = config['activation']
                    print('Building an ASN-' + activation + ' output layer as layer ' + str(l_test) + ' in model_test')
                    # This triggers that the spikes are being integrated into S and evaluated as a softmax at
                    # every time_step
                    x = ASN_1D(nodes, use_bias=True, activation=activation, last_layer=True, mf=mf,
                               h_scaling=h_scaling, input_layer=input_layer)(x)
                    if spike_counting == True:
                        if l_test == 1:
                            spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                        else:
                            s = SpikeSum(h_scaling=h_scaling)(x)
                            spike_counter = Concatenate()([spike_counter, s])

                    l_train = l_train + 1
                    l_test = l_test +  1
                    model_match.append(l_train)

                elif model_training.layers[l_train + 1].__class__.__name__ == 'BatchNormalization':
                    if model_training.layers[l_train + 2].__class__.__name__ == 'ASNTransfer':
                        print('Building joint Dense-BN-ASN Layer as layer ' + str(l_test) + ' in model_test')
                        nodes = model_training.layers[l_train].output_shape[1]
                        x = ASN_1D(nodes, use_bias=True, mf=mf, h_scaling=h_scaling,
                                   input_layer=input_layer)(x)

                        if spike_counting == True:
                            if l_test == 1:
                                spike_counter = SpikeSum(h_scaling=h_scaling)(x)
                            else:
                                s = SpikeSum(h_scaling=h_scaling)(x)
                                spike_counter = Concatenate()([spike_counter, s])

                        l_train = l_train + 3
                        l_test = l_test + 1
                        model_match.append(l_train)
                    else:
                        print('Building time-distributed Dense-BN Layer as layer ' + str(l_test) + ' in model_test')
                        nodes = model_training.layers[l_train].output_shape[1]
                        x = TimeDistributed(Dense(nodes, activation='linear', use_bias=False))(x)

                        l_train = l_train + 2
                        l_test = l_test + 1
                        model_match.append(l_train)
                else:
                    nodes = model_training.layers[l_train].output_shape[1]
                    if model_training.layers[l_train].input.name.startswith(identity_name):  # Identity
                        print('Building time-distributed Dense as layer ' + str(l_test) + ' in model_test for identity')
                        identity = TimeDistributed(Dense(nodes, activation='linear', use_bias=False))(identity)

                        # Update the identity name
                        identity_name = model_training.layers[i].name

                    else:
                        print('Building time-distributed Dense as layer ' + str(l_test) + ' in model_test')  # Main
                        x = TimeDistributed(Dense(nodes, activation='linear', use_bias=False))(x)

                    l_train = l_train + 1
                    l_test = l_test + 1
                    model_match.append(l_train)

            elif mode == 'Flatten':
                print('Building time-distributed ' + mode + ' as layer ' + str(l_test) + ' in model_test')
                x = TimeDistributed(Flatten())(x)
                l_train = l_train + 1
                l_test = l_test + 1
                model_match.append(l_train)

            elif mode in ['Dropout', 'GaussianDropout','GaussianNoise','UniformNoise']:
                l_train = l_train + 1
                print('Dropout or noise is not used during testing and therefore not included in model_test')

            elif mode == 'Activation':
                if len(model_training.layers) == l_train + 1:  # Make a softmax output layer for the last layer
                    nodes = model_training.layers[l_train].output_shape[1]
                    config = model_training.layers[l_train].get_config()
                    activation = config['activation']
                    print('Building an ASN-' + activation + ' output layer as layer ' + str(l_test) + ' in model_test')
                    # This triggers that the spikes are being integrated into S and evaluated as a softmax at
                    # every time_step
                    x = ASN_1D(nodes, use_bias=True, activation=activation, last_layer=True, mf=mf,
                               h_scaling=h_scaling, input_layer=input_layer)(x)

                    l_train = l_train + 1
                    l_test = l_test + 1
                    model_match.append(l_train)
                else:
                    print('Activation layers that are not an ASNTransfer function can only be used as an output of the '
                          'network.')

        if len(model_training.layers[i]._outbound_nodes) > 1:
            identity_name = model_training.layers[i].name
            # If a layer has more than 1 outgoing connection it is an identity mapping for a resnet block
            identity = x
    if spike_counting == True:
        if attention_applied == True:
            model_test = Model(inputs=[input, input_att], outputs=[x, spike_counter])
        else:
            model_test = Model(inputs=input, outputs=[x, spike_counter])
    else:
        if attention_applied == True:
            model_test = Model(inputs=[input, input_att], outputs=x)
        else:
            model_test = Model(inputs=input, outputs=x)
    # This is here because we are using an offset in the counting of model_train
    model_match = np.array(model_match) - 1
    return model_test, model_match


def convert_weights(model_training, model_test, h, model_match):

    print('Transferring weights ...')
    # Layer counters for both models
    if model_training.layers[0].__class__.__name__ == 'InputLayer':
        l_train = 1
    else:
        l_train = 0

    if model_test.layers[0].__class__.__name__ == 'InputLayer':
        l_test = 1  # Since the first two layers are the input layers

    bias = []

    for i in np.arange(0, len(model_training.layers)):
        mode = model_training.layers[i].__class__.__name__
        print('Evaluating layer ' + str(i) + ' - ' + mode)
        print(model_training.layers[i].input_shape)
        print(model_test.layers[l_test].__class__.__name__)
        if model_test.layers[l_test].__class__.__name__ in ['InputLayer']:
            l_test = l_test+1
        # Skip all spike counting related layers, which are uniquely having only 2 dimensions
        while len(model_test.layers[l_test].output_shape) == 2:
            l_test = l_test + 1
            print(model_test.layers[l_test].__class__.__name__)

        if i < l_train:
            print('Information from layer ' + str(i) + ' (' + mode + ') has already been integrated')
        else:
            if mode == 'BatchNormalization':
                if model_training.layers[l_train + 1].__class__.__name__ == 'ASNTransfer':  # Test for combination BN & ASN

                    print('Loading in BN weights from Model_training layer ' + str(l_train))
                    BN_weights = model_training.layers[l_train].get_weights()
                    weights = model_test.layers[l_test].get_weights()
                    # Replace 1-weights with the identity matrix between the channel dimensions, such that every channel
                    # is only connected to itself in the next layer
                    weights[0] = np.identity(weights[0].shape[-1]).reshape(weights[0].shape)  # keep the shape the same.

                    if len(bias) > 0:  # collect the biases
                        integrated_bias = 0
                        for b in bias:
                            integrated_bias = np.array(b) + integrated_bias

                        weights[1] = integrated_bias
                        bias = []

                    if l_test == 1:  # if this is the first layer then don't scale by h, only apply BN
                        scaled_weights = normalize_weights(weights, BN_weights=BN_weights)
                    else:
                        scaled_weights = normalize_weights(weights, BN_weights=BN_weights, h=h)  # also scaling the kernel with h

                    model_test.layers[l_test].set_weights(scaled_weights)  # Assign new weights based on trained BN params
                    l_test = l_test + 1

                    if model_training.layers[l_train + 2].__class__.__name__ in ['AveragePooling2D', 'MaxPooling2D']:  # Test for combination BN & ASN
                        l_train = l_train + 3

                    else:
                        l_train = l_train + 2  # because 2 layers were added

                else:
                    print('BN weights will be integrated with other layers')

                    l_train = l_train + 1
                    l_test = l_test

            elif mode == 'ASNTransfer':
                print('Building ' + mode + ' as layer ' + str(l_test) + ' in model_test')
                weights = model_test.layers[l_test].get_weights()
                # Replace 1-weights with the identity matrix between the channel dimensions, such that every channel
                # is only connected to itself in the next layer
                weights[0] = np.identity(weights[0].shape[-1]).reshape(weights[0].shape)  # keep the shape the same.
                if len(bias) > 0: # collect the biases
                    integrated_bias = 0
                    for b in bias:
                        integrated_bias = np.array(b) + integrated_bias

                    weights[1]= integrated_bias
                    bias = []
                scaled_weights = normalize_weights(weights, h=h) # scale by h to comply with transfer function
                model_test.layers[l_test].set_weights(scaled_weights)  # Assign new weights

                l_train = l_train + 1
                l_test = l_test + 1

            elif mode == 'Conv2D':
                if (model_training.layers[l_train + 1].__class__.__name__ == 'BatchNormalization') & \
                        (model_training.layers[l_train]._outbound_nodes[0].outbound_layer.name.startswith(
                        'batch_norm')): # This second part tests the connection.:
                    if model_training.layers[l_train + 2].__class__.__name__ == 'ASNTransfer':
                        # assemble trained conv and BN weights
                        print('Loading in Conv2d weights from Model_training layer ' + str(l_train))
                        weights = model_training.layers[l_train].get_weights()
                        print('Loading in BN weights from Model_training layer ' + str(l_train + 1))
                        BN_weights = model_training.layers[l_train + 1].get_weights()
                        print('Integrating BN weights from Model_training layer ' + str(l_train + 1))

                        if (l_test == 1) | (model_training.layers[l_train + 2].__class__.__name__ != 'ASNTransfer'):  # if this is the first layer then don't scale by h, only apply BN
                            print('Not scaling by h!')
                            scaled_weights = normalize_weights(weights, BN_weights=BN_weights)
                        else:
                            scaled_weights = normalize_weights(weights, BN_weights=BN_weights, h=h)  # also scaling the kernel with h

                        # Transfer trained weights
                        model_test.layers[l_test].set_weights(scaled_weights)
                        l_test = l_test + 1

                        if model_training.layers[l_train + 3].__class__.__name__ in ['MaxPooling2D','AveragePooling2D']:

                            l_train = l_train + 4  # because now 4 layers were combined

                        elif model_training.layers[l_train + 3].__class__.__name__ in ['GaussianDropout','Dropout']:
                            if model_training.layers[l_train + 4].__class__.__name__ in ['MaxPooling2D','AveragePooling2D']:
                                l_train = l_train + 5  # because now 5 layers were combined
                            else:
                                l_train = l_train + 4
                        else:
                            l_train = l_train + 3  # because now 3 layers were combined

                    else:
                        print('Loading in Conv2d weights from Model_training layer ' + str(l_train))
                        weights = model_training.layers[l_train].get_weights()

                        print('Loading in BN weights from Model_training layer ' + str(l_train + 1))
                        BN_weights = model_training.layers[l_train + 1].get_weights()

                        scaled_weights = normalize_weights(weights, BN_weights=BN_weights)

                        bias.append(scaled_weights[1])
                        model_test.layers[l_test].set_weights([scaled_weights[0]])

                        l_train = l_train+2
                        l_test = l_test+1

                elif model_training.layers[l_train + 1].__class__.__name__ == 'ASNTransfer':

                    print('Loading in Conv2d weights from Model_training layer ' + str(l_train))
                    weights = model_training.layers[l_train].get_weights()
                    weights[0] = weights[0] * h  # scale by h to comply with transfer function
                    model_test.layers[l_test].set_weights(weights)

                    l_train = l_train + 2
                    l_test = l_test + 1


                else:

                    # Identity & main branches sometimes get scrambeld up during the assembling of the model,
                    # therefore we have to check with which time-dist Conv we are dealing

                    print('Loading in Conv2d weights from Model_training layer ' + str(l_train))
                    weights = model_training.layers[l_train].get_weights()

                    if model_training.layers[l_train]._outbound_nodes[0].outbound_layer.name.startswith(
                            'batch_norm'):
                        print('Found some BN weights!')
                        BN_weights = model_training.get_layer(
                            model_training.layers[l_train]._outbound_nodes[0].outbound_layer.name).get_weights()
                        weights = normalize_weights(weights, BN_weights=BN_weights)

                    bias.append(weights[1])
                    if model_test.layers[l_test].get_weights()[0].shape != weights[0].shape:
                        model_test.layers[l_test+1].set_weights([weights[0]])
                        weights2 = model_training.layers[l_train+1].get_weights()
                        bias.append(weights2[1])

                        model_test.layers[l_test].set_weights([weights2[0]])
                        l_train = l_train + 2
                        l_test = l_test + 2

                    else:
                        model_test.layers[l_test].set_weights([weights[0]])
                        l_train = l_train + 1
                        l_test = l_test + 1

            elif mode in ['MaxPooling2D', 'AveragePooling2D', 'Add', 'Subtract', 'Flatten']:

                l_train = l_train + 1
                l_test = l_test + 1


            elif mode == 'Dense':

                if len(model_training.layers) == l_train + 1:  # Make a softmax output layer for the last layer
                    weights = model_training.layers[l_train].get_weights()
                    model_test.layers[l_test].set_weights(weights)

                elif model_training.layers[l_train + 1].__class__.__name__ == 'BatchNormalization':
                    if model_training.layers[l_train + 2].__class__.__name__ == 'ASNTransfer':
                        print('Loading in Dense weights from Model_training layer ' + str(l_train))
                        weights = model_training.layers[l_train].get_weights()
                        BN_weights = model_training.layers[l_train + 1].get_weights()
                        print('Integrating BN weights from Model_training layer ' + str(l_train + 1))

                        scaled_weights = normalize_weights(weights, BN_weights=BN_weights,
                                                           h=h)  # also scaling the kernel with h
                        # Transfer trained weights
                        model_test.layers[l_test].set_weights(scaled_weights)

                        l_train = l_train + 3
                        l_test = l_test + 1
                    else:
                        print('Loading in Dense weights from Model_training layer ' + str(l_train))
                        weights = model_training.layers[l_train].get_weights()
                        BN_weights = model_training.layers[l_train + 1].get_weights()
                        print('Integrating BN weights from Model_training layer ' + str(l_train + 1))

                        scaled_weights = normalize_weights(weights, BN_weights)

                        bias.append(scaled_weights[1])
                        # Transfer trained weights
                        model_test.layers[l_test].set_weights([scaled_weights[0]])

                        l_train = l_train + 2
                        l_test = l_test + 1
                else:
                    print('Loading in Dense weights from Model_training layer ' + str(l_train))
                    weights = model_training.layers[l_train].get_weights()
                    bias.append(weights[1])
                    # Transfer trained weights
                    model_test.layers[l_test].set_weights([weights[0]])

                    l_train = l_train + 1
                    l_test = l_test + 1

            elif mode == 'Activation':

                print('Building ' + mode + ' as layer ' + str(l_test) + ' in model_test')
                weights = model_test.layers[l_test].get_weights()
                # Replace 1-weights with the identity matrix between the channel dimensions, such that every channel
                # is only connected to itself in the next layer
                weights[0] = np.identity(weights[0].shape[-1]).reshape(weights[0].shape)  # keep the shape the same.
                if len(bias) > 0:  # collect the biases
                    integrated_bias = 0
                    for b in bias:
                        if model_training.layers[l_train-1].__class__.__name__ == 'Add':
                            integrated_bias = np.array(b) + integrated_bias
                        elif model_training.layers[l_train-1].__class__.__name__ == 'Subtract':
                            if integrated_bias == 0:
                                integrated_bias = np.array(b)
                            else:
                                integrated_bias = integrated_bias - np.array(b)

                    weights[1] = integrated_bias
                    bias = []
                scaled_weights = normalize_weights(weights, h=h)  # scale by h to comply with transfer function
                model_test.layers[l_test].set_weights(scaled_weights)  # Assign new weights

                l_train = l_train + 1
                l_test = l_test + 1

            elif mode in ['Dropout', 'GaussianDropout','GaussianNoise','UniformNoise']:
                l_train = l_train + 1
                print('Dropout or Noise is not used during testing and therefore not included in model_test')


    return model_test


def convert_model(model_training, time_steps, *args, mf=0.1, h_weightscaling=True, skip=[], skip_value=0.06,
                   spike_counting=False):
    """This function translates between the training architecture and the ASN architecture
    :parameter
    model_training: analog Keras model with trained weights
    time_steps for the digital/spiking model conversion
    *args: attn_param, dicts with parameter for attentional manipulation
    mf: precision for spike generation. Default: 0.1
    h_weightscaling: weights of kernels will be scaled by h to account for adherence to the transfer function based on mf
    resnet: whether given model is a resnet?


    :returns
    model_test: Converted sDNN architecture with trained weights from model_training
    model_match: Layer correspondence between analog & spiking network.

    Last updated 18.04.19
    """
    if len(args) == 0:
        attn_param = set_model_attn_param(model_training)
    else:
        attn_param = set_model_attn_param(model_training, args)

    if h_weightscaling == True:
        h = normalize_transfer(mf)  # for adherence to the transfer function
        h_scaling = None
    else:
        h = None
        h_scaling = True
        print('Assuming that spike production will be scaled by h')

    model_test, model_match = convert_architecture(model_training, time_steps, mf, h_scaling, attn_param, skip=skip,
                                                   skip_value=skip_value, spike_counting=spike_counting)

    model_test = convert_weights(model_training, model_test, h, model_match)

    print('Conversion complete: ')
    model_test.summary()
    return model_test, model_match
