

from keras.layers import Layer
import tensorflow as tf
from keras import backend as K

from asn.conversion.utils import normalize_transfer

class ASNTransfer(Layer):
    '''Parametric Adaptive Spiking Neuron Transfer function:
    Used for training based on Zambrano et al. 2017

    Please note: The threshold (cutoff for negative values, cf. reluval) can be changed to e.g. 0.1. This is
    advantageous for input with low activations, which risk to be in the range of the transfer function that the spiking
    layers cannot cover (values below ca. 0.1). Yet not that this introduces a problem for backprob, since it makes the
    transfer function discontinuous.
    07.11.2018
    '''
    def __init__(self, m_f=0.1, threshold=0.0, **kwargs):
        self.supports_masking = True

        self.m_f = m_f
        self.tau_gamma = 15.0
        self.tau_eta = 50.0
        self.theta0 = m_f
        self.h = normalize_transfer(m_f) # 0.1244 for m_f= 0.1,
        self.threshold = threshold  # for ReLU cut-off

        self.c1 = 2*self.m_f*self.tau_gamma*self.tau_gamma
        self.c2 = 2*self.theta0*self.tau_eta*self.tau_gamma
        self.c3 = self.tau_gamma*(self.m_f*self.tau_gamma + 2*(self.m_f + 1)*self.tau_eta)
        self.c4 = self.theta0*self.tau_gamma*self.tau_eta + self.theta0*self.tau_eta*self.tau_eta

        self.c0 = self.h/(K.exp((self.c1*0.5*self.theta0 + self.c2)/(self.c3*0.5*self.theta0 + self.c4)) - 1)

        super(ASNTransfer, self).__init__(**kwargs)

    def build(self, input_shape):

        super(ASNTransfer, self).build(input_shape)

    def call(self, x, mask=None):

        reluval = x * K.cast(x >= self.threshold, K.floatx())
        act_val = self.h/(K.exp((self.c1*reluval + self.c2)/(self.c3*reluval + self.c4)) - 1.) - self.c0 + self.h/2.0

        r_act_val = K.relu(act_val)

        return r_act_val



