# -*- coding: utf-8 -*-
from math import factorial

import numpy as np
from scipy import fftpack
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

try:
    from lmfit import Model

    _has_lmfit = True
except ImportError:
    _has_lmfit = False


from pysprint.utils import (
    find_nearest,
    _handle_input,
    unpack_lmfit,
    fourier_interpolate,
    transform_cf_params_to_dispersion,
    transform_lmfit_params_to_dispersion,
    plot_phase,
)

from pysprint.core.functions import *


__all__ = [
    "min_max_method",
    "spp_method",
    "cff_method",
    "fft_method",
    "cut_gaussian",
    "ifft_method",
    "args_comp",
]


_fit_config = {
    1: poly1,
    2: poly2,
    3: poly3, 
    4: poly4,
    5: poly5
}


_cosfit_config = {
    1: cos_fit1,
    2: cos_fit2,
    3: cos_fit3,
    4: cos_fit4,
    5: cos_fit5,
}


def min_max_method(
    x,
    y,
    ref,
    sam,
    ref_point,
    maxx=None,
    minx=None,
    fit_order=5,
    show_graph=False,
):
    """Calculates the dispersion with minimum-maximum method

    Parameters
    ----------

    x: array-like
    x-axis data

    y: array-like
    y-axis data

    ref, sam: array-like
    reference and sample arm spectra evaluated at x

    ref_point: float
    the reference point to calculate order

    maxx and minx: array-like, optional
    the accepted minimal and maximal x values (if you want to manually pass)

    fit_order: int, optional
    degree of polynomial to fit data [1, 5]

    show_graph: bool, optional
    if True plot the calculated phase

    Returns
    -------

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    dispersion_std: array-like
    [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

    fit_report: lmfit report
    """
    if fit_order not in range(6):
        raise ValueError("fit order must be in [1, 5]")

    x, y = _handle_input(x, y, ref, sam)
    if maxx is None:
        maxInd = argrelextrema(y, np.greater)
        maxx = x[maxInd]
    if minx is None:
        minInd = argrelextrema(y, np.less)
        minx = x[minInd]

    _, ref_index = find_nearest(x, ref_point)

    max_freq = x[ref_index] - maxx
    min_freq = x[ref_index] - minx

    neg_freq = np.sort(
        np.append(max_freq[max_freq < 0], min_freq[min_freq < 0])
    )[::-1]
    pos_freq = np.sort(
        np.append(max_freq[max_freq > 0], min_freq[min_freq > 0])
    )

    if len(neg_freq) == 0 and len(pos_freq) == 0:
        raise ValueError("No extremal points found.")

    pos_values = np.pi * np.arange(1, len(pos_freq) + 1)
    neg_values = np.pi * np.arange(1, len(neg_freq) + 1)

    x_s = np.append(pos_freq, neg_freq)
    y_s = np.append(pos_values, neg_values)

    # FIXME: Do we even need this?
    # Yes, we do. This generates a prettier plot.
    idx = np.argsort(x_s)
    full_x, full_y = x_s[idx], y_s[idx]

    _function = _fit_config[fit_order]

    if _has_lmfit:
        fitmodel = Model(_function)
        pars = fitmodel.make_params(
            **{f"b{i}": 1 for i in range(fit_order + 1)}
        )
        result = fitmodel.fit(full_y, x=full_x, params=pars)
    else:
        popt, pcov = curve_fit(_function, full_x, full_y, maxfev=8000)

    if _has_lmfit:
        dispersion, dispersion_std = transform_lmfit_params_to_dispersion(
            *unpack_lmfit(result.params.items()), drop_first=True, dof=1
        )
        fit_report = result.fit_report()
    else:
        dispersion, dispersion_std = transform_cf_params_to_dispersion(
            popt, drop_first=True
        )
        fit_report = (
            "To display detailed results," " you must have `lmfit` installed."
        )
    if show_graph:
        try:
            plot_phase(
                full_x,
                full_y,
                bf=result.best_fit,
                bf_fallback=_function(full_x, *popt),
                window_title="Min-max method fitted",
            )
        except UnboundLocalError:

            class result:
                def best_fit():
                    return None

            plot_phase(
                full_x,
                full_y,
                bf=result.best_fit,
                bf_fallback=_function(full_x, *popt),
                window_title="Min-max method fitted",
            )

    return dispersion, dispersion_std, fit_report


def spp_method(delays, omegas, ref_point=0, fit_order=4):
    """
    Calculates the dispersion from SPP's positions and delays.

    Parameters
    ----------

    delays: array-like
    The delay values in fs exactly where omegas taken.

    omegas: array-like
    The angular frequency values where SPP's are located

    fit_order: int
    order of polynomial to fit the given data

    Returns
    -------
    omegas: array-like
    x axis data

    delays: array-like
    y axis data

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    dispersion_std: array-like
    [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

    bf: array-like
    best fitting curve for plotting
    """
    if fit_order not in range(5):
        raise ValueError("fit order must be in [1, 4]")

    omegas = np.asarray(omegas).astype(np.float64)

    delays = np.asarray(delays).astype(np.float64)

    if not len(delays) == len(omegas):
        raise ValueError(
            f"data shapes are different: {delays.shape} & {omegas.shape}"
        )

    idx = np.argsort(omegas)
    omegas, delays = omegas[idx], delays[idx]
    omegas -= ref_point

    _function = _fit_config[fit_order]

    try:
        if _has_lmfit:
            fitmodel = Model(_function)
            pars = fitmodel.make_params(
                **{f"b{i}": 1 for i in range(fit_order + 1)}
            )
            result = fitmodel.fit(delays, x=omegas, params=pars)

            dispersion, dispersion_std = transform_lmfit_params_to_dispersion(
                *unpack_lmfit(result.params.items()), drop_first=False, dof=0
            )
            bf = result.best_fit
        else:
            popt, pcov = curve_fit(_function, omegas, delays, maxfev=8000)
            dispersion, dispersion_std = transform_cf_params_to_dispersion(
                popt, drop_first=False
            )
            bf = _function(omegas, *popt)
        return omegas, delays, dispersion, dispersion_std, bf

    except Exception as e:
        raise e  # this should be deleted..


def cff_method(
    x, y, ref, sam, ref_point=0, p0=[1, 1, 1, 1, 1, 1, 1, 1], maxtries=8000
):
    """
    Phase modulated cosine function fit method.

    Parameters
    ----------

    x: array-like
    x-axis data

    y: array-like
    y-axis data

    ref, sam: array-like
    the reference and sample arm spectra evaluated at x

    p0: array-like
    the initial parameters for fitting

    Returns
    -------

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    bf: array-like
    best fitting curve

    """

    x, y = _handle_input(x, y, ref, sam)

    try:
        orderhelper = np.max(np.flatnonzero(p0)) - 2

        p0 = np.trim_zeros(p0, "b")

        _funct = _cosfit_config[orderhelper]

        popt, pcov = curve_fit(_funct, x - ref_point, y, p0, maxfev=maxtries)

        dispersion = np.zeros_like(popt)[:-3]
        for num in range(len(popt) - 3):
            dispersion[num] = popt[num + 3] * factorial(num + 1)
        return dispersion, _funct(x - ref_point, *popt)
    except RuntimeError:
        raise ValueError(
            f"""Max tries ({maxtries}) reached..
                             Parameters could not be estimated."""
        )


def fft_method(x, y):
    """Perfoms FFT on data

    Parameters
    ----------

    x: array-like
    the x-axis data

    y: array-like
    the y-axis data

    Returns
    -------
    xf: array-like
    the transformed x data

    yf: array-like
    transformed y data
    """
    yf = fftpack.fft(y)
    xf = np.linspace(x[0], x[-1], len(x))
    return xf, yf


def gaussian_window(t, tau, fwhm, order):
    """
    Returns a simple gaussian window of given parameters evaulated at t.

    Parameters
    ----------
    t: array-like
    input array to perform window on

    tau: float
    center of gaussian window

    fwhm: float
    FWHM of given gaussian

    order: float
    order of gaussian window. If not even it's incremented by 1.

    Returns
    -------
    array : array-like
    nth order gaussian window with params above

    """
    if order % 2 == 1:
        order += 1
    std = fwhm / (2 * (np.log(2) ** (1 / order)))
    return np.exp(-((t - tau) ** order) / (std ** order))


def cut_gaussian(x, y, spike, fwhm, win_order):
    """
    Applies gaussian window with the given params.

    Parameters
    ----------
    x: array-like
    x-axis data

    y: array-like
    y-axis data

    spike: float
    center of gaussian window

    fwhm: float
    Full width at half max

    Returns
    -------

    y: array-like
    the windowed y values
    """
    y *= gaussian_window(x, tau=spike, fwhm=fwhm, order=win_order)
    return y


def ifft_method(x, y, interpolate=True):
    """
    Perfoms IFFT on data

    Parameters
    ----------

    x: array-like
    the x-axis data

    y: array-like
    the y-axis data

    interpolate: bool
    if True perform a linear interpolation on dataset before transforming

    Returns
    -------
    xf: array-like
    the transformed x data

    yf: array-like
    transformed y data

    """
    N = len(x)
    if interpolate:
        x, y = fourier_interpolate(x, y)
    xf = np.fft.fftfreq(N, d=(x[1] - x[0]) / (2 * np.pi))
    yf = np.fft.ifft(y)
    return xf, yf


def args_comp(x, y, ref_point=0, fit_order=5, show_graph=False):
    """
    Calculates the phase of complex dataset then unwrap by changing
    deltas between values to 2*pi complement. At the end, fit a
    polynomial curve to determine dispersion coeffs.

    Parameters
    ----------

    x: array-like
    the x-axis data

    y: array-like
    the y-axis data

    ref_point: float
    the reference point to calculate order

    fit_order: int
    degree of polynomial to fit data [1, 5]

    show_graph: bool
    if True return a plot with the results

    Returns
    -------

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    dispersion_std: array-like
    [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

    fit_report: lmfit report
    """
    if fit_order not in range(6):
        raise ValueError("fit order must be in [1, 5]")

    x -= ref_point
    y = np.unwrap(np.angle(y), axis=0)

    _function = _fit_config[fit_order]

    if _has_lmfit:
        fitmodel = Model(_function)
        pars = fitmodel.make_params(
            **{f"b{i}": 1 for i in range(fit_order + 1)}
        )
        result = fitmodel.fit(y, x=x, params=pars)
    else:
        popt, pcov = curve_fit(_function, x, y, maxfev=8000)

    if _has_lmfit:
        dispersion, dispersion_std = transform_lmfit_params_to_dispersion(
            *unpack_lmfit(result.params.items()), drop_first=True, dof=1
        )
        fit_report = result.fit_report()
    else:
        dispersion, dispersion_std = transform_cf_params_to_dispersion(
            popt, drop_first=True, dof=1
        )
        fit_report = (
            "To display detailed results," " you must have `lmfit` installed."
        )
    if show_graph:
        try:
            plot_phase(
                x,
                y,
                result.best_fit,
                bf_fallback=_function(x, *popt),
                window_title="Phase",
            )
        except UnboundLocalError:

            class result:
                def best_fit():
                    return None

            plot_phase(
                x,
                y,
                result.best_fit,
                bf_fallback=_function(x, *popt),
                window_title="Phase",
            )

    return -dispersion, dispersion_std, fit_report
