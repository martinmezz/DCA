import numpy as np

class ModifiedHyperbolicDecline:
    def __init__(self, qi, di, b, t_transition):
        self.qi = qi  # Initial rate
        self.di = di  # Initial decline rate
        self.b = b    # Decline exponent
        self.t_transition = t_transition  # Time to transition from hyperbolic to exponential

    def predict(self, t):
        """Predict production rate at time t based on the modified hyperbolic model."""
        if [t.to_list() < self.t_transition]:
            return self.qi / ((1 + self.b * self.di * t) ** (1/self.b))
        else:
            q_transition = self.qi / ((1 + self.b * self.di * self.t_transition) ** (1/self.b))
            di_exp = self.di / (self.t_transition ** self.b)  # Convert to exponential decline
            return q_transition * np.exp(-di_exp * (t - self.t_transition))
