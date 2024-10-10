import numpy as np

def mean_squared_error(actual, predicted):
    """Calculate Mean Squared Error between actual and predicted values."""
    return np.mean((actual - predicted) ** 2)

def r_squared(actual, predicted):
    """Calculate the R-squared value."""
    residual_sum_of_squares = np.sum((actual - predicted) ** 2)
    total_sum_of_squares = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)
