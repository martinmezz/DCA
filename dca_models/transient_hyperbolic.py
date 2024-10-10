class TransientHyperbolicDecline:
    def __init__(self, qi, di, b, tau):
        self.qi = qi  # Initial rate
        self.di = di  # Decline rate
        self.b = b    # Decline exponent
        self.tau = tau  # Transient time constant

    def predict(self, t):
        """Predict production rate at time t using transient hyperbolic decline model."""
        return self.qi / ((1 + (t / self.tau) * self.di) ** (1/self.b))
