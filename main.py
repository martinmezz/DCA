import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from dca_models.power_law import PowerLawDecline
from dca_models.stretched_exponential import StretchedExponentialDecline
from dca_models.arps import ArpsDecline
from dca_models.modified_hyperbolic import ModifiedHyperbolicDecline
from dca_models.transient_hyperbolic import TransientHyperbolicDecline
from utils.model_selection import fit_model

# Load the data
data = load_data('data/mock_production_data.json')

# Select a well's data (e.g., well_id == 1)
well_data = data[data['well_id'] == 570]

# Fit Arps, Power Law and Stretched Exponential models
time = well_data['production_time']
oil_rate = well_data['oil_rate']

# Fit Arps Decline model
arps_params = fit_model(ArpsDecline,time, oil_rate,qi=5000, di=0.02, b=0.5)
arps_model = ArpsDecline(*arps_params)

# Fit Modified Hyperbolic Decline model
mod_hyperbolic_params = fit_model(ModifiedHyperbolicDecline, time, oil_rate,qi=5000, di=0.02, b=0.5, t_transition=500)
mod_hyperbolic = ModifiedHyperbolicDecline(*mod_hyperbolic_params)

# Fit Transient Hyperbolic Decline model
trans_hyperbolic_params = fit_model(TransientHyperbolicDecline, time, oil_rate,qi=5000, di=0.02, b=0.7, tau=500)
trans_hyperbolic_model = TransientHyperbolicDecline(*trans_hyperbolic_params)

# Fit Power Law model
power_law_params = fit_model(PowerLawDecline, time, oil_rate, qi=5000, n=0.5)
power_law_model = PowerLawDecline(*power_law_params)

# Fit Stretched Exponential model
stretched_exp_params = fit_model(StretchedExponentialDecline, time, oil_rate, qi=5000, tau=100, beta=0.5)
stretched_exp_model = StretchedExponentialDecline(*stretched_exp_params)

tiempo = np.linspace(1,2000,50)

# Plot the actual vs. predicted data
plt.plot(time, oil_rate, label='Actual Data',ls= '--',marker='o')
plt.plot(time, arps_model.predict(time), label='Arps Fit')
plt.plot(time, mod_hyperbolic.predict(time), label='Modified Hyperbolic Fit', marker='o')
plt.plot(time, trans_hyperbolic_model.predict(time), label='Transient Hyperbolic Fit')
plt.plot(time, power_law_model.predict(time), label='Power Law Fit')
plt.plot(tiempo, stretched_exp_model.predict(tiempo), label='Stretched Exponential Fit')
plt.xlabel('Time (days)')
plt.ylabel('Oil Production Rate (bbl/day)')
plt.legend()
plt.savefig('output/best_fit_model_plot.png')
plt.show()
