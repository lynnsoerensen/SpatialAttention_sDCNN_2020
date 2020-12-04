import numpy as np

class ASN:
    """ Adaptive spiking neuron class, double-checked on Jan 24, 2018 """

    def __init__(self, mf=0.1, bias=0):
        # Params
        # membrane filter
        self.tau_phi = 2.5
        self.dPhi = np.exp(-1 / self.tau_phi)
        # threshold decay filter
        self.tau_gamma = 15.0
        self.dGamma = np.exp(-1 / self.tau_gamma)
        # refractory decay  filter
        self.tau_eta = 50.0
        self.dEta = np.exp(-1 / self.tau_eta)
        self.dBeta = self.dEta

        self.m_f = mf  # **2 would be the old matlab code
        self.theta0 = self.m_f  # Resting threshold

        self.S_bias = bias
        self.S = self.S_bias  # filtered activation, initialized with bias
        self.S_dyn = 0
        self.theta = self.theta0  # Start value of threshold
        self.theta_dyn = 0  # dynamic part of the threshold
        self.S_hat = 0  # refractory response, internal approximation

        self.current_next = 0  # incoming current in next neuron
        self.S_next = 0  # and filtered by the membrane potential.
        self.I = 0
        self.spike = 0

    def update(self, current, spike_train=True):
        """inject current for one moment at a time"""
        # Membrane filter
        if spike_train:
            self.I = self.I * self.dBeta + current
        else:
            self.I = current
        self.S_dyn = (1 - self.dPhi) * self.I + self.dPhi * self.S_dyn
        self.S = self.S_bias + self.S_dyn
        # Decay
        self.S_hat = self.S_hat * self.dEta
        self.current_next = self.current_next * self.dEta
        # Spike?
        if self.S - self.S_hat > 0.5 * self.theta:
            self.spike = 1  # Code spike

            # Update refractory response
            self.S_hat = self.S_hat + self.theta

            # Update threshold
            # self.theta_dyn = self.theta_dyn + self.m_f*self.theta/self.theta0 #based on the matlab code
            self.theta_dyn = self.theta_dyn + self.m_f * self.theta  # adaptive part based on the paper

            self.current_next = self.current_next + 1
        else:
            self.spike = 0

        # Decay
        self.theta_dyn = self.theta_dyn * self.dGamma
        self.theta = self.theta0 + self.theta_dyn

        # Signal in next neuron
        self.S_next = (1 - self.dPhi) * self.current_next + self.dPhi * self.S_next

    def call(self, input, spike_train=True, mf=0.1, bias=0):
        timesteps = input.shape[1]
        batch_size = input.shape[0]
        S = np.zeros(input.shape)
        S_next = np.zeros(input.shape)
        spikes = np.zeros(input.shape)
        S_hat = np.zeros(input.shape)
        theta = np.zeros(input.shape)

        for b in range(batch_size):
            self.__init__(mf=mf, bias=bias)
            for t in range(timesteps):  # loop over timesteps
                self.update(input[b, t, :], spike_train=spike_train)
                S[b, t, 0] = self.S
                S_next[b, t, 0] = self.S_next
                spikes[b, t, 0] = self.spike
                S_hat[b, t, 0] = self.S_hat
                theta[b, t, 0] = self.theta

        return S_next, spikes, S, S_hat, theta

