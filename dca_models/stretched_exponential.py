import numpy as np

class StretchedExponentialDecline:
    def __init__(self, qi, tau, beta):
        self.qi = qi    # Initial rate
        self.tau = tau  # Characteristic time
        self.beta = beta  # Stretching exponent

    def predict(self, t):
        """Predict production rate at time t (days)"""
        return self.qi * np.exp(- (t / self.tau) ** self.beta)
