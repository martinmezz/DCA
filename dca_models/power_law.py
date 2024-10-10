import numpy as np

class PowerLawDecline:
    def __init__(self, qi, n, t0=1):
        self.qi = qi  # Initial rate
        self.n = n    # Decline exponent
        self.t0 = t0  # Reference time

    def predict(self, t):
        """Predict production rate at time t (days)"""
        return self.qi * (t / self.t0) ** (-self.n)
