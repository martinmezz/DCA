from scipy.optimize import curve_fit
import numpy as np

def fit_model(model, time, production):
    """Fit the given model to the production data."""
    popt, _ = curve_fit(model.predict, time, production)
    return popt  # Optimal parameters (qi, di, b for Arps)


def fit_model(model_class, time, production, **initial_guess):
    """Fit the model to the production data."""
    def model_func(t, *params):
        model = model_class(*params)
        return model.predict(t)

    popt, _ = curve_fit(model_func, time, production, p0=list(initial_guess.values()))
    return popt  # Optimal parameters
