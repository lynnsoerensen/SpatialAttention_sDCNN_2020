import tensorflow as tf
import math
import keras.backend as K


def makeGaussian(space, center, width, height=None):
    """
    adapted from Heeger & Reynolds (2009) - The Normalization model of Attention from MATLAB documentation:
    gaussian = makeGaussian(space,center,width,[height])

    This is a function creates gaussian centered at "center",
    over the values defined by vector "space".

    width is the standard deviation of the gaussian

    height, if specified is the height of the peak of the gaussian.
    Otherwise, it is scaled to unit volume
    """


    pi = tf.constant(math.pi)
    center_mat = tf.reshape(tf.keras.backend.repeat(tf.expand_dims(center, axis=0), tf.shape(space)[0]),
                            [tf.shape(space)[0], tf.shape(center)[0]])
    space_mat = tf.cast(tf.reshape(tf.keras.backend.repeat(tf.expand_dims(space, axis=1), tf.shape(center)[0]),
                                   [tf.shape(space)[0], tf.shape(center)[0]]), dtype='float32')
    Z = 1.0 / (width * (2.0 * pi) ** (1 / 2))
    gaussian = Z * tf.exp(-0.5 * (space_mat - center_mat) ** 2 / width ** 2)

    if height is not None:
        c = height * width * tf.sqrt(2 * pi)
        gaussian = c * gaussian
    return gaussian


def attnGain_2D(x1,x2,Ax1_params, Ax2_params, attn_param):
    """Function generates 2-d kernel with attnGain (rows, cols)

    x1: Vertical definition of space, centered around 0
    x2: Horizontal definition of space, centered around 0
    attn_param contains the following fields:
    Ax1: Vertical center of attention field (rows) in proportions to x1
    Ax2: Horizontal center of attention field (columns)
    AxWidth1: vertical extent/width  attention field
    AxWidth2: horizontal extent/width  attention field
    Apeak: peak amplitude of attention field
    Abase: baseline of attention field for unattended locations/features
    Ashape: either 'oval' or 'cross'

    This is adapted from the NMA code by Reynolds & Heeger, 2009
    """
    # If no attentional weighting is applied, initialize the entire field with ones.
    # This corresponds to spreading the attentional resources equally over space and features.
    #if all([Ax1_params is None, Ax2_params is None]):
    if all([attn_param['AxWidth1'] is None, attn_param['AxWidth2'] is None]):
        input_shape = (tf.shape(x1)[0], tf.shape(x2)[0])
        attnGain = tf.ones((1,) + input_shape)

    else:
        # %% This first part is to ensure that Ax1 and Ax2 always have the same length
        if attn_param['AxWidth1'] is None:
            Ax1 = tf.ones(tf.shape(Ax2_params))
            attnGainX1 = tf.ones((tf.shape(Ax1_params)[0], tf.shape(x1)[0], 1))

        else:
            if attn_param['AxWidth2'] is None:
                Ax1 = tf.ones(tf.shape(Ax1_params)) * Ax1_params
            else:
                Ax1 = tf.ones(tf.shape(Ax2_params)) * Ax1_params

            # Gaussian kernel with the centre of the attentional field along the rows.
            # Note that the height is normalized to 1.
            attnGainX1 = makeGaussian(tf.cast(x1,dtype='float32'),
                                      Ax1, tf.div(float(attn_param["AxWidth1"]), tf.cast(tf.shape(x1)[0], dtype='float32')),
                                      height=1)

            if attn_param["Ashape"] is "cross":
                attnGainX1 = (attn_param["Apeak"] - attn_param["Abase"]) * attnGainX1 + attn_param["Abase"]

        if attn_param['AxWidth2'] is None:
            Ax2 = tf.ones(tf.shape(Ax1_params))
            attnGainX2 = tf.ones((tf.shape(Ax2)[0], 1, tf.shape(x2)[0]))

        else:
            if attn_param['AxWidth1'] is None:
                Ax2 = tf.ones(tf.shape(Ax2_params)) * Ax2_params
            else:
                Ax2 = tf.ones(tf.shape(Ax1_params)) * Ax2_params

            attnGainX2 = makeGaussian(tf.cast(x2, dtype='float32'),
                                      Ax2,
                                      tf.div(float(attn_param["AxWidth2"]), tf.cast(tf.shape(x2)[0], dtype='float32')),
                                      height=1)

            if attn_param["Ashape"] is "cross":
                attnGainX2 = (attn_param["Apeak"] - attn_param["Abase"]) * attnGainX2 + attn_param["Abase"]

        # Multiply the two spatial vectors
        attnGainX1 = tf.expand_dims(tf.transpose(attnGainX1), axis = 2)
        attnGainX2 = tf.expand_dims(tf.transpose(attnGainX2), axis=1)
        attnGain = tf.matmul(attnGainX1, attnGainX2, )
        if attn_param["exponent"]:
            attnGain=attnGain**attn_param["exponent"]

        # Normalize with regard to maximum peak and the baseline of the attentional field.
        attnGain = (attn_param["Apeak"] - attn_param["Abase"]) * attnGain + attn_param["Abase"]

    return attnGain

def makeGaussian_np(space, center, width, height=None):
    """from original MATLAB documentation:
    gaussian = makeGaussian(space,center,width,[height])

    This is a function creates gaussian centered at "center",
    over the values defined by vector "space".

    width is the standard deviation of the gaussian

    height, if specified is the height of the peak of the gaussian.
    Otherwise, it is scaled to unit volume
    """
    import numpy as np
    from scipy.stats import norm

    gaussian = norm.pdf(space, center, width)
    if height is not None:
        gaussian = height * width * np.sqrt(2 * np.pi) * gaussian
    return gaussian


def attnGain_2D_np(x1, x2, attn_param):

    """Function generates 2-d kernel with attnGain (rows, cols)

        x1: Vertical definition of space, centered around 0
        x2: Horizontal definition of space, centered around 0
        attn_param contains the following fields:
        Ax1: Vertical center of attention field (rows) in proportions to x1
        Ax2: Horizontal center of attention field (columns)
        AxWidth1: vertical extent/width  attention field
        AxWidth2: horizontal extent/width  attention field
        Apeak: peak amplitude of attention field
        Abase: baseline of attention field for unattended locations/features
        Ashape: either 'oval' or 'cross'
    """
    import numpy as np

    # If no attentional weighting is applied, initialize the entire field with ones.
    # This corresponds to spreading the attentional resources equally over space and features.
    if all([attn_param["Ax1"] is None, attn_param["Ax2"] is None]):
        input_shape = (x1.shape[0], x2.shape[0])
        attnGain = np.ones(input_shape)
    else:
        # Otherwise check which of the features has biased attention
        if attn_param["Ax1"] is None:
            attnGainX1 = np.ones((x1.shape[0]))
        else:
            # Gaussian kernel with the centre of the attentional field along the rows.
            # Note that the height is normalized to 1.
            Ax1 = attn_param["extent_x1"] * attn_param["Ax1"]
            attnGainX1 = makeGaussian_np(x1, Ax1, attn_param["AxWidth1"], height = 1)

            if attn_param["Ashape"] is "cross":
                attnGainX1 = (attn_param["Apeak"] - attn_param["Abase"]) * attnGainX1 + attn_param["Abase"]
        # Reshape the vector for later multiplication
        attnGainX1 = attnGainX1.reshape((attnGainX1.shape[0], 1))
        # The same steps are applied to Ax2(rows):
        if attn_param["Ax2"] is None:
            attnGainX2 = np.ones(x2.shape[0])
        else:
            Ax2 = attn_param["extent_x2"] * attn_param["Ax2"]
            attnGainX2 = makeGaussian_np(x2, Ax2, attn_param["AxWidth2"], height = 1)
            if attn_param["Ashape"] is "cross":
                attnGainX2 = (attn_param["Apeak"] - attn_param["Abase"]) * attnGainX2 + attn_param["Abase"]
        attnGainX2 = attnGainX2.reshape((1, attnGainX2.shape[0]))

        # Multiply the two spatial vectors
        attnGain = np.multiply(attnGainX1, attnGainX2)

        if attn_param["exponent"]:
            attnGain = attnGain ** attn_param["exponent"]

        # Normalize with regard to maximum peak and the baseline of the attentional field.
        attnGain = (attn_param["Apeak"] - attn_param["Abase"]) * attnGain + attn_param["Abase"]

    return attnGain


def normalize_precisionMap(mf_mat, mf, lowest_precision=0.06):
    """
    This function ensures that the precision maps has the same sum irrespective of the position of the attentional kernel.
    It also corrects for the assignment of unrealistic thresholds.

    :param mf_mat: R-weighted precision map
    :param mf: baseline mf
    :param lowest_precision: lowest mf to avoid oversampling

    :return: normalized mf_mat
    """
    #K.shape
    output_shape = K.shape(mf_mat)
    h_axis, w_axis, c_axis = 1, 2, 3
    normalizing_factor = tf.reduce_sum(tf.ones((output_shape[h_axis], output_shape[w_axis], output_shape[c_axis]), dtype='float32') * mf *
                                       10) / tf.reduce_sum(mf_mat[:,:,:,:] * 10, axis=[1, 2, 3])  # The 10 are added to avoid rounding imprecisions
    mf_mat = mf_mat * normalizing_factor[:, tf.newaxis, tf.newaxis, tf.newaxis]

    # This protects from assigning too high thresholds.
    oversampling = tf.cast(mf_mat < lowest_precision, tf.bool)
    mf_mat = tf.where(oversampling, tf.ones(tf.shape(mf_mat)) * lowest_precision, mf_mat)

    return mf_mat