
from __future__ import print_function

import numpy as np
import os
import keras
from keras.layers import Dense
from keras.layers.convolutional import _Conv
from keras.engine import Layer
from keras.layers import InputSpec
from keras.utils import conv_utils
from keras import backend as K
from keras import initializers, activations, regularizers, constraints
from asn.conversion.utils import normalize_transfer
from asn.attention.attend import attend_tf
from asn.attention.utils import normalize_precisionMap
import tensorflow as tf


class ASN_2D(_Conv):
    """ Layer with integrated adaptive spiking neuron layer
    The input has 5 dimensions (batch,time,feature1,feature2,channels)
    If filter & kernel_size are specified it acts as a 2D convolutional layer.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(time,128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

        pool_mode, pool_size, pool_stride, pool_padding: If provided, this layer adds a pooling operation.
        mf: resting threshold
        input_layer: If true, the neuron model will expect a current instead of a spike train
        last_layer: If true, the neuron model will give out a current instead of a spike train.
        h_scaling: if spike trains should be scaled or not. Alternative is that the weights are pre-scaled.
        attn_param: Has no function here, only for compatibility reasons.
    # Input shape
        5D tensor with shape:
        `(batch, time_steps,channels, rows, cols)`

    # Output shape
        5D tensor with shape:
        `(batch, time_steps, filters, new_rows, new_cols)`


    """

    def __init__(self, filters=(1),
                 kernel_size=(1,1),
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 rank=2, pool_mode=None, pool_size=None,
                 pool_stride=None, pool_padding='valid',
                 mf=None,
                 input_layer=False, last_layer=False,
                 h_scaling=None, attn_param= {"layer_idx": None},
                 **kwargs):
        self.supports_masking = True
        self.input_layer = input_layer
        self.last_layer = last_layer
        self.h_scaling = h_scaling

        # Params
        # membrane filter
        self.tau_phi=2.5

        self.dPhi = K.constant(np.exp(-1 / self.tau_phi))
        # threshold decay filter
        self.tau_gamma = 15.0
        self.dGamma = K.constant(np.exp(-1 / self.tau_gamma))
        # refractory decay  filter
        self.tau_eta = 50.0
        self.dEta = K.constant(np.exp(-1 / self.tau_eta))

        self.tau_beta = self.tau_eta
        self.dBeta = self.dEta

        if mf == None:
            self.mf = K.constant(0.1)  # Resting threshold
        else:
            self.mf = mf

        if self.h_scaling:
            self.h = normalize_transfer(self.mf)
        else:
            self.h = 1

        self.attn_param = attn_param

        # Pooling Params
        self.pool_mode = pool_mode
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.pool_stride = pool_stride

        super(ASN_2D, self).__init__(rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self,input_shape):
        #output_shape = list(input_shape)
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        if self.data_format == 'channels_first':
            h_axis, w_axis = 3, 4
        else:
            h_axis, w_axis = 2, 3

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        out_height = conv_utils.conv_output_length(height,kernel_h,self.padding,stride_h)

        out_width = conv_utils.conv_output_length(width,kernel_w,self.padding,stride_w)

        if self.data_format == 'channels_first':
            conv_output_shape = (batch_size, time_steps, self.filters, out_height, out_width)

        else:
            conv_output_shape = (batch_size, time_steps, out_height, out_width, self.filters)


        if self.pool_size != None:
            pool_out_height = conv_utils.conv_output_length(out_height,self.pool_size[0],self.pool_padding , self.pool_stride[0])
            pool_out_width = conv_utils.conv_output_length(out_width,self.pool_size[1],self.pool_padding , self.pool_stride[1])
            if self.data_format == 'channels_first':

                pool_output_shape = (batch_size,time_steps,self.filters, pool_out_height, pool_out_width)
            else:
                pool_output_shape = (batch_size,time_steps, pool_out_height, pool_out_width, self.filters)

            output_shape = pool_output_shape
        else:
            output_shape = conv_output_shape

        return output_shape



    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError('Inputs should have rank ' +
                             str(5) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        output_shape = self.compute_output_shape(input_shape)
        if (self.attn_param['layer_idx'] is not None) & (self.h_scaling == None):
            raise ValueError('It is not possible to not scale the spikes by h (h_scaling == False) '
                             'and at the same time apply attention parameters')

        # Set input spec.
        self.input_spec = InputSpec(ndim=len(input_shape), axes={channel_axis: input_dim})
        self.built = True

    def update(self, current, states):
        """inject current for one moment in time at once"""
        # states: [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, S_pool, I]
        theta = states[0]
        theta0_mat = states[1]
        theta_dyn = states[2]
        S_hat = states[3]
        S_bias = states[4]
        S_dyn = states[5]
        #S = states[6]
        S_pool = states[6]
        I = states[7]

        # Apply convolution
        current = K.conv2d(current, self.kernel, strides=self.strides, padding=self.padding,
                               data_format=self.data_format)

        # Membrane filter
        if self.input_layer == True:  # in the case when the input to the neuron is already a current, e.g. pixel values
            I = current

        else: # when the input is a spiking sequence
            I = I * self.dBeta + current

        # S_pool is needed because it has still more units and we need to keep those on record so as to be able to
        # maxpool

        S_pool = (1 - self.dPhi) * I + self.dPhi * S_pool

        if self.pool_size != None:

            S_dyn = K.pool2d(S_pool, self.pool_size, strides=self.pool_stride, padding=self.pool_padding,
                             data_format=self.data_format, pool_mode=self.pool_mode)
        else:
            S_dyn = S_pool # If no pooling is applied this reduces to being the same as S_dyn

        S = S_bias + S_dyn

        # Decay
        S_hat = S_hat * self.dEta

        # Spike?
        spike = tf.cast(S - S_hat > 0.5 * theta, tf.float32)  # Code spike

        # Update refractory response
        # self.S_hat = self.S_hat + self.theta
        S_hat = S_hat + tf.multiply(theta, spike)

        # Update threshold
        theta_dyn = theta_dyn + tf.multiply(tf.multiply(theta, spike), self.mf)

        # Decay
        theta_dyn = theta_dyn * self.dGamma
        theta = theta0_mat + theta_dyn

        if self.last_layer == True:
            out = S  # for the last layer give out the S instead of spikes
        else:
            out = spike * self.h_mat  # if it is a spike, scale by h

        return out, [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S_pool, I]


    def call(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        time_steps = K.int_shape(inputs)[1]

        if self.data_format == 'channels_first':
            h_axis, w_axis = 3, 4
        else:
            h_axis, w_axis = 2, 3

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        conv_out_height = conv_utils.conv_output_length(height,kernel_h,self.padding,stride_h)
        conv_out_width= conv_utils.conv_output_length(width,kernel_w,self.padding,stride_w)

        if self.data_format == 'channels_first':
            conv_output_shape = (batch_size,self.filters,conv_out_height,conv_out_width)

        else:
            conv_output_shape = (batch_size,conv_out_height,conv_out_width,self.filters)

        # Maxpooling output if required
        if self.pool_size != None:
            pool_out_height = conv_utils.conv_output_length(conv_out_height,self.pool_size[0],self.pool_padding , self.pool_stride[0])
            pool_out_width = conv_utils.conv_output_length(conv_out_width,self.pool_size[1],self.pool_padding , self.pool_stride[1])
            if self.data_format == 'channels_first':

                pool_output_shape = (batch_size,self.filters, pool_out_height, pool_out_width)

            else:
                pool_output_shape = (batch_size, pool_out_height, pool_out_width, self.filters)

            output_shape = pool_output_shape

            # To determine MaxPooling values
            I = tf.zeros(conv_output_shape)  # current inside of the neuron

        else:
            output_shape = conv_output_shape
            I = tf.zeros(output_shape)  # current inside of the neuron

        if self.use_bias:
            # filtered activation. Note that here the bias from training is used as an initial state of the neuron.
            S_bias = tf.multiply(tf.ones(output_shape), self.bias)
        else:
            S_bias = tf.zeros(output_shape)

        #  Loop over all time points for one sample
        #  Preallocate neuron tensors

        if self.h_scaling:
            self.h_mat = tf.cast(normalize_transfer(self.mf * tf.ones(output_shape[2:len(output_shape)], dtype='float32'), mode='tf'), dtype='float32')
        else:
            self.h_mat = tf.ones(output_shape[2:len(output_shape)], dtype='float32')

        S_pool = tf.zeros(conv_output_shape)
        S = S_bias
        S_dyn = tf.zeros(output_shape)
        theta0_mat = tf.multiply(tf.ones(output_shape), self.mf)
        theta = tf.multiply(tf.ones(output_shape), self.mf)  # Start value of thresehold
        theta_dyn = tf.zeros(output_shape)  # dynamic part of the threshold
        S_hat = tf.zeros(output_shape)  # refractory response, internal approximation

        # Loop over all time points
        last_output, outputs, states = K.rnn(self.update,
                                             inputs,
                                             [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S_pool, I],
                                             unroll=False,
                                             input_length=K.int_shape(inputs)[1])
        return outputs

    def get_config(self):
        config = {
            'pool_mode':  self.pool_mode,
            'pool_size': self.pool_size,
            'pool_stride': self.pool_stride,
            'pool_padding': self.pool_padding,
            'attn_param': self.attn_param
        }
        base_config = super(ASN_2D, self).get_config()
        base_config.pop('dilation_rate')
        return dict(list(base_config.items()) + list(config.items()))


#%%
class ASN_2D_attention(_Conv):
    """ Layer with integrated adaptive spiking neuron layer with
    The first input has 5 dimensions (batch,time,feature1,feature2,channels), the second are two coordinates for the
    placement of the attention field.
    If filter & kernel_size are specified, it acts as a 2D convolutional layer, otherwise it can act as a dense layer.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(time,128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

        pool_mode, pool_size, pool_stride, pool_padding: If provided, this layer adds a pooling operation.
        mf: resting threshold
        input_layer: If true, the neuron model will expect a current instead of a spike train
        last_layer: If true, the neuron model will give out a current instead of a spike train.
        h_scaling: if spike trains should be scaled or not. Alternative is that the weights are pre-scaled.
        attn_param: specified with asn.attention.atten_param.py

    # Input shape
        5D tensor with shape:
        `(batch, time_steps, rows, cols, channels)`
        2D list with shape:
        `(batch, coordinates (2 -> row, col))
    # Output shape
        or 5D tensor with shape:
        `(batch, time_stepsnew_rows, new_cols, filters)` .


    """

    def __init__(self, filters=(1),
                 kernel_size=(1,1),
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 rank=2, pool_mode=None, pool_size=None,
                 pool_stride=None, pool_padding='valid',
                 mf=None,
                 input_layer=False, last_layer=False,
                 h_scaling=None, attn_param= {"layer_idx": None},
                 **kwargs):
        self.supports_masking = True
        self.input_layer = input_layer
        self.last_layer = last_layer
        self.h_scaling = h_scaling
        # Params
        # membrane filter
        self.tau_phi=2.5
        self.dPhi = K.constant(np.exp(-1 / self.tau_phi))
        # threshold decay filter
        self.tau_gamma = 15.0
        self.dGamma = K.constant(np.exp(-1 / self.tau_gamma))
        # refractory decay  filter
        self.tau_eta = 50.0
        self.dEta = K.constant(np.exp(-1 / self.tau_eta))

        self.tau_beta = self.tau_eta
        self.dBeta = self.dEta

        if mf == None:
            self.mf = K.constant(0.1)  # Resting threshold
        else:
            self.mf = mf

        if self.h_scaling:
            self.h = normalize_transfer(self.mf)
        else:
            self.h = 1

        self.attn_param = attn_param

        # Pooling Params
        self.pool_mode = pool_mode
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.pool_stride = pool_stride

        super(ASN_2D_attention, self).__init__(rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = [InputSpec(ndim=5), InputSpec(ndim=2)]  # 5

    def compute_output_shape(self,input_shape):

        batch_size = input_shape[0][0]
        time_steps = input_shape[0][1]

        if self.data_format == 'channels_first':
            h_axis, w_axis = 3, 4
        else:
            h_axis, w_axis = 2, 3

        height, width = input_shape[0][h_axis], input_shape[0][w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        out_height = conv_utils.conv_output_length(height,kernel_h,self.padding,stride_h)

        out_width = conv_utils.conv_output_length(width,kernel_w,self.padding,stride_w)

        if self.data_format == 'channels_first':
            conv_output_shape = (batch_size, time_steps, self.filters, out_height, out_width)

        else:
            conv_output_shape = (batch_size, time_steps, out_height, out_width, self.filters)

        if self.pool_size != None:
            pool_out_height = conv_utils.conv_output_length(out_height,self.pool_size[0],self.pool_padding , self.pool_stride[0])
            pool_out_width = conv_utils.conv_output_length(out_width,self.pool_size[1],self.pool_padding , self.pool_stride[1])
            if self.data_format == 'channels_first':

                pool_output_shape = (batch_size,time_steps,self.filters, pool_out_height, pool_out_width)
            else:
                pool_output_shape = (batch_size,time_steps, pool_out_height, pool_out_width, self.filters)

            output_shape = pool_output_shape
        else:
            output_shape = conv_output_shape

        return output_shape


    def build(self, input_shape):
        if len(input_shape[0]) != 5:
            raise ValueError('Inputs should have rank ' +
                             str(5) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        output_shape = self.compute_output_shape(input_shape)

        # Set input spec.
        self.input_spec = [InputSpec(ndim=len(input_shape[0]), axes={channel_axis: input_dim}), InputSpec(ndim=len(input_shape[1]))]
        self.built = True

    def update(self, current, states):
        """inject current for one moment in time at once"""
        # states: [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, S_pool, I]
        theta = states[0]
        theta0_mat = states[1]
        theta_dyn = states[2]
        S_hat = states[3]
        S_bias = states[4]
        S_dyn = states[5]
        S = states[6]
        S_pool = states[7]
        I = states[8]

        # Apply convolution
        current = K.conv2d(current, self.kernel, strides=self.strides, padding=self.padding,
                               data_format=self.data_format)

        # Membrane filter
        if self.input_layer == True:
            # in the case when the input to the neuron is already a current, e.g. pixel values
            I = current

        else:  # when the input is a spiking sequence
            I = I * self.dBeta + current

        # S_pool is needed because it has still more units and we need to keep those on record so as to be able to
        # maxpool
        S_pool = (1 - self.dPhi) * I * self.gain + self.dPhi * S_pool

        if self.pool_size != None:

            S_dyn = K.pool2d(S_pool, self.pool_size, strides=self.pool_stride, padding=self.pool_padding,
                             data_format=self.data_format, pool_mode=self.pool_mode)
        else:
            S_dyn = S_pool  # If no pooling is applied this reduces to being the same as S_dyn

        S = S_bias + S_dyn

        # Decay
        S_hat = S_hat * self.dEta

        # Spike?
        spike = tf.cast(S - S_hat > 0.5 * theta, tf.float32)  # Code spike

        # Update refractory response
        S_hat = S_hat + tf.multiply(theta, spike)

        # Update threshold
        theta_dyn = theta_dyn + tf.multiply(tf.multiply(theta, spike), self.mf_mat)

        # Decay
        theta_dyn = theta_dyn * self.dGamma
        theta = theta0_mat + theta_dyn

        if self.last_layer == True:
            out = S  # for the last layer give out the S instead of spikes
        else:
            out = spike * self.h_mat  # if it is a spike scale by h

        return out, [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, S_pool, I]


    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('An attention layer should be called '
                             'on a list of inputs.')

        input_shape = K.shape(inputs[0])
        batch_size = input_shape[0]

        if self.data_format == 'channels_first':
            h_axis, w_axis = 3, 4
        else:
            h_axis, w_axis, c_axis = 2, 3, 4

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        conv_out_height = conv_utils.conv_output_length(height,kernel_h,self.padding,stride_h)
        conv_out_width= conv_utils.conv_output_length(width,kernel_w,self.padding,stride_w)

        if self.data_format == 'channels_first':
            conv_output_shape = (batch_size,self.filters,conv_out_height,conv_out_width)

        else:
            conv_output_shape = (batch_size,conv_out_height,conv_out_width,self.filters)

        #  Maxpooling output if required
        if self.pool_size != None:
            pool_out_height = conv_utils.conv_output_length(conv_out_height,self.pool_size[0],self.pool_padding , self.pool_stride[0])
            pool_out_width = conv_utils.conv_output_length(conv_out_width,self.pool_size[1],self.pool_padding , self.pool_stride[1])
            if self.data_format == 'channels_first':

                pool_output_shape = (batch_size,self.filters, pool_out_height, pool_out_width)

            else:
                pool_output_shape = (batch_size, pool_out_height, pool_out_width, self.filters)

            output_shape = pool_output_shape

            #  To determine MaxPooling values
            I = tf.zeros(conv_output_shape) #current inside of the neuron
        else:
            output_shape = conv_output_shape
            I = tf.zeros(output_shape) #current inside of the neuron

        #  Loop over all time points for one sample
        #  Preallocate neuron tensors
        if self.use_bias:
            S_bias = tf.multiply(tf.ones(output_shape), self.bias) # filtered activation. Note that here the bias from training is used as an initial state of the neuron.
        else:
            S_bias = tf.zeros(output_shape)

        S_pool = tf.zeros(conv_output_shape)
        S = S_bias
        S_dyn = tf.zeros(output_shape)

        # Determine the attentional reweighting for every image in the batch
        R = attend_tf(output_shape, self.data_format, self.attn_param, Ax1=inputs[1][:,0], Ax2=inputs[1][:,1])

        # This can introduce a precision modulation
        if self.attn_param['Precision'] == True:
            self.mf_mat = self.mf - R * self.mf
            # This ensures that the sum of all mfs is 0 per feature map.
            self.mf_mat = normalize_precisionMap(self.mf_mat, self.mf)
        else:
            self.mf_mat = self.mf * tf.ones(output_shape, dtype='float32')

        # This can introduce a input gain modulation if input Gain is > 0
        self.gain = R*self.attn_param['InputGain'] + 1

        # This can introduce an output gain modulation if output Gain is > 0
        self.h_mat = normalize_transfer(self.mf_mat, mode='tf') * (R*self.attn_param['OutputGain'] + 1)

        theta0_mat = self.mf_mat
        theta = self.mf_mat  # Start value of threshold

        theta_dyn = tf.zeros(output_shape)  # dynamic part of the threshold
        S_hat = tf.zeros(output_shape)  # refractory response, internal approximation

        # Loop over all time points
        last_output, outputs, states = K.rnn(self.update,
                                             inputs[0],
                                             [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, S_pool, I],
                                             unroll=False,
                                             input_length=input_shape[1])
        return outputs

    def get_config(self):
        config = {
            'pool_mode':  self.pool_mode,
            'pool_size': self.pool_size,
            'pool_stride': self.pool_stride,
            'pool_padding': self.pool_padding,
            'attn_param': self.attn_param
        }
        base_config = super(ASN_2D_attention, self).get_config()
        base_config.pop('dilation_rate')
        return dict(list(base_config.items()) + list(config.items()))


class ASN_1D(Dense):
    """ Adaptive spiking neuron class

    Adapted from a regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).

        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).


        mf: resting threshold
        input_layer: If true, the neuron model will expect a current instead of a spike train
        last_layer: If true, the neuron model will give out a current instead of a spike train.
        h_scaling: if spike trains should be scaled or not. Alternative is that the weights are pre-scaled.

        last_layer: Key command for a read-out ASN neuron
        # Input shape
            nD tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.
        # Output shape
            nD tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='ones',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_layer=False,
                 last_layer=False,
                 mf=None,
                 h_scaling=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ASN_1D, self).__init__(units, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True
        self.input_layer = input_layer
        self.last_layer = last_layer

        # Params for ASN neuron
        # membrane filter
        self.tau_phi = 2.5
        if self.last_layer == True:
            self.tau_phi = 50.  # Longer temporal integration for last layer
        self.dPhi = K.constant(np.exp(-1 / self.tau_phi))
        # threshold decay filter
        self.tau_gamma = 15.0
        self.dGamma = K.constant(np.exp(-1 / self.tau_gamma))
        # refractory decay  filter
        self.tau_eta = 50.0
        self.dEta = K.constant(np.exp(-1 / self.tau_eta))
        self.dBeta = self.dEta

        if mf == None:
            self.mf = 0.1  # Resting threshold
        else:
            self.mf = mf

        self.theta0 = self.mf

        if h_scaling:
            self.h = normalize_transfer(self.mf)
        else:
            self.h = 1

    def build(self, input_shape):
        assert len(input_shape) >= 3
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=3, axes={-1: input_dim})
        self.built = True

    def update(self, current, states):
        """inject current for one moment in time at once"""
        # states: [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, I]
        theta = states[0]
        theta0_mat = states[1]
        theta_dyn = states[2]
        S_hat = states[3]
        S_bias = states[4]
        S_dyn = states[5]
        S = states[6]
        I = states[7]

        # Apply dense weights
        current = tf.matmul(current, self.kernel)

        # Membrane filter
        if self.input_layer == True:  # in the case when the input to the neuron is already a current, e.g. pixel values
            I = current

        else:  # when the input is a spiking sequence
            I = I * self.dBeta + current

        # Membrane filter
        S_dyn = (1 - self.dPhi) * I + self.dPhi * S_dyn
        S = S_bias + S_dyn

        # Decay
        S_hat = S_hat * self.dEta

        # Spike?
        spike = tf.cast(S - S_hat > 0.5 * theta, tf.float32)  # Code spike

        # Update refractory response
        S_hat = S_hat + tf.multiply(theta, spike)

        # Update threshold
        theta_dyn = theta_dyn + tf.multiply(tf.multiply(theta, spike), self.mf)

        # Decay
        theta_dyn = theta_dyn * self.dGamma
        theta = theta0_mat + theta_dyn

        if self.last_layer == True:
            out = self.activation(S)  # for the last layer give out the S instead of spikes
        else:
            out = spike * self.h  # if it is a spike scale by h

        return out, [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, I]

    def call(self, inputs, mask=None):

        batch_size = K.shape(inputs)[0]

        # Preallocate tensors and reset for every example
        theta = tf.ones((batch_size, self.units)) * self.theta0  # Start value of thresehold
        theta0_mat = tf.ones((batch_size, self.units)) * self.theta0
        theta_dyn = tf.zeros((batch_size, self.units))  # dynamic part of the threshold
        S_hat = tf.zeros((batch_size, self.units))  # refractory response, internal approximation

        if self.use_bias:
            # filtered activation. Note that here the bias from training is used as an initial state of the neuron.
            S_bias = tf.multiply(tf.ones((batch_size, self.units)),
                                 self.bias)
        else:
            S_bias = tf.zeros((batch_size, self.units))

        S_dyn = tf.zeros((batch_size, self.units))  # dynamic part of the activation
        S = S_bias  # as initial state for activation
        I = tf.zeros((batch_size, self.units))

        # Loop over all time points
        last_output, outputs, states = K.rnn(self.update,
                                             inputs,
                                             [theta, theta0_mat, theta_dyn, S_hat, S_bias, S_dyn, S, I],
                                             unroll=False,
                                             input_length=K.int_shape(inputs)[1])

        return outputs

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


class SpikeSum(Layer):
    """
    This layer sums a tensor of spikes, only batch dimension is maintained.
    This layer is used to keep track of the total spike spendings per image.

    Takes any spiking layer as input.

    added on 17.01.2020

    """

    def __init__(self, h_scaling=False, **kwargs):
        self.h_scaling = h_scaling
        super(SpikeSum, self).__init__(**kwargs)

    def call(self, x):
        shape = K.int_shape(x)
        if self.h_scaling == True:
            x = tf.cast(x > 0, tf.float32)
        return tf.reduce_sum(x, np.arange(1, len(shape)))[:, tf.newaxis]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
