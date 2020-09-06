from math import factorial

import numpy as np

from pysprint.utils import pad_with_trailing_zeros


__all__ = [
    "transform_cf_params_to_dispersion",
    "transform_lmfit_params_to_dispersion",
]


def transform_cf_params_to_dispersion(popt, drop_first=True, dof=0):
    if drop_first:
        popt = popt[1:]

    _disp = popt

    for idx in range(len(popt)):
        _disp[idx] = popt[idx] * factorial(idx + dof)

    _disp = pad_with_trailing_zeros(_disp, 6)
    return _disp, np.zeros(6)


def transform_lmfit_params_to_dispersion(popt, popt_std, drop_first=True, dof=1):
    if drop_first:
        popt, popt_std = popt[1:], popt_std[1:]

    _disp, _disp_std = popt, popt_std

    for idx in range(len(popt)):
        _disp[idx] = popt[idx] * factorial(idx + dof)
        if popt_std[idx] is not None:
            _disp_std[idx] = popt_std[idx] * factorial(idx + dof)
        else:
            _disp_std[idx] = 0
    _disp = pad_with_trailing_zeros(popt, 6)
    _disp_std = pad_with_trailing_zeros(popt_std, 6)
    return _disp, _disp_std
