import tensorflow as tf
from asn.attention.utils import attnGain_2D


def attend_tf(output_shape, data_format, attn_param, Ax1=None, Ax2=None):

    """ This function is to create an attention map based on a spatial redistribution for the rows and cols within a
    feature map

    """
    # Check formatting
    if data_format == 'channels_first':
        h_axis, w_axis, c_axis = 3, 4, 2
    elif data_format == 'channels_last':
        h_axis, w_axis, c_axis = 1, 2, 3
    else:
        raise ValueError('Data_format is not understood.')

    sampling = tf.cast(output_shape[h_axis], dtype='float32')
    # Sample to the right centered space
    x1 = tf.range(tf.div(1.0, 2.0 * sampling), 1, delta=tf.div(1.0, sampling))
    x2 = tf.range(tf.div(1.0, 2.0 * sampling), 1, delta=tf.div(1.0, sampling))

    # The precision weighting is identical for all feature maps in a given layer.
    # Accordingly, the prec ision-weighing is performed within a feature map and the replicated in depth.
    # It assumes that if no attention is applied precision is spread out equally
    E = attnGain_2D(x1, x2, Ax1, Ax2, attn_param)

    R = (E / tf.reduce_mean(E, axis=[1, 2], keepdims=True)) - 1

    # round R to 1e-5 precision
    multiplier = tf.constant(10 ** 4, dtype='float32')
    R = tf.round(R * multiplier) / multiplier

    # and project it in depth
    R_deep = R[:,:,:,tf.newaxis] * tf.ones((tf.shape(R)[0],output_shape[h_axis], output_shape[w_axis], output_shape[c_axis]))

    return R_deep


