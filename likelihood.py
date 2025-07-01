import numpy as np
from scipy.interpolate import interp1d

def condensed_likelihood(theta_data, r_data, sig_data, theta_model, r_model):

    mask = ~np.isnan(r_model)
    interp_r = interp1d(theta_model[mask], r_model[mask], kind='linear', bounds_error=False, fill_value='extrapolate')

    return np.nanmean((interp_r(theta_data) - r_data)**2 / sig_data**2)

def condensed_bias(theta_data, r_data, sig_data, theta_model, r_model):

    mask = ~np.isnan(r_model)
    interp_r = interp1d(theta_model[mask], r_model[mask], kind='linear', bounds_error=False, fill_value='extrapolate')

    return np.nanmean((interp_r(theta_data) - r_data) / sig_data)