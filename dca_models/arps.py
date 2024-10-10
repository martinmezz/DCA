import numpy as np

class ArpsDecline:
    def __init__(self, qi, di, b):
        self.qi = qi  # Initial rate
        self.di = di  # Decline rate
        self.b = b    # Decline exponent

    def predict(self, t):
        """Predict the production rate at time t (months)"""
        return self.qi / ((1 + self.b * self.di * t) ** (1/self.b))
