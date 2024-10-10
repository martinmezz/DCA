from scipy.optimize import curve_fit
import numpy as np
from utils.metrics import mean_squared_error

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

def select_best_model(models, time, production, initial_guesses):
    """Fit multiple models and select the one with the lowest MSE."""
    best_model = None
    lowest_mse = float('inf')
    
    for model_class, initial_guess in models.items():
        # Fit model
        params = fit_model(model_class, time, production, **initial_guess)
        model = model_class(*params)
        
        # Calculate MSE
        predictions = model.predict(time)
        mse = mean_squared_error(production, predictions)
        
        if mse < lowest_mse:
            lowest_mse = mse
            best_model = model
    
    return best_model, lowest_mse