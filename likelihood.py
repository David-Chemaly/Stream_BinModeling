import numpy as np
from scipy.interpolate import interp1d

def condensed_likelihood(dict_data, dict_model, mask_data=True):

    mask = ~np.isnan(dict_model['r_bin'])
    interp_r = interp1d(dict_model['theta_bin'][mask], dict_model['r_bin'][mask], kind='linear', bounds_error=False, fill_value='extrapolate')

    return np.nanmean((interp_r(dict_data['theta_bin'])[mask_data] - dict_data['r_bin'][mask_data])**2 / dict_data['r_sig_bin'][mask_data]**2)


def condensed_bias(dict_data, dict_model, mask_data=True):

    mask = ~np.isnan(dict_model['r_bin'])
    interp_r = interp1d(dict_model['theta_bin'][mask], dict_model['r_bin'][mask], kind='linear', bounds_error=False, fill_value='extrapolate')

    return np.nanmean((interp_r(dict_data['theta_bin'])[mask_data] - dict_data['r_bin'][mask_data]) / dict_data['r_sig_bin'][mask_data])
