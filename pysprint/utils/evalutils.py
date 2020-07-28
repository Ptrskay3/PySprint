from math import factorial

import numpy as np
import matplotlib.pyplot as plt

from pysprint.utils import pad_with_trailing_zeros


__all__ = [
    "transform_cf_params_to_dispersion",
    "transform_lmfit_params_to_dispersion",
    "plot_phase",
]


def transform_cf_params_to_dispersion(popt, drop_first=True, dof=0):
    if drop_first:
        popt = popt[1:]

    _disp = popt

    for idx in range(len(popt)):
        _disp[idx] = popt[idx] * factorial(idx + dof)

    _disp = pad_with_trailing_zeros(_disp, 5)
    return _disp, np.zeros(5)


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
    _disp = pad_with_trailing_zeros(popt, 5)
    _disp_std = pad_with_trailing_zeros(popt_std, 5)
    return _disp, _disp_std


# DEPRECATE THIS
def plot_phase(
    x,
    y,
    bf,
    *,
    bf_fallback=None,
    figsize=(7, 7),
    window_title=None,
    show_labels=True,
    dataset_color="k",
    fit_color="r",
    grid=True
):

    if window_title is None:
        window_title = "Phase"

    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title(window_title)
    plt.plot(x, y, c=dataset_color, label="Dataset")
    try:
        plt.plot(x, bf, c=fit_color, linestyle="dashed", label="Best fit")
    except ValueError:
        if bf_fallback is None:
            pass
        else:
            plt.plot(
                x, bf_fallback, c=fit_color, linestyle="dashed", label="Best fit",
            )
    if show_labels:
        plt.legend()
    if grid:
        plt.grid()
    plt.show()
