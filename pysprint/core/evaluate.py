import logging
import numbers
from math import factorial
from typing import List, Union, Tuple

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

from pysprint.core.functions import _fit_config, _cosfit_config

Num = Union[int, float]
NumericLike = Union[Num, Union[List, np.ndarray]]

logger = logging.getLogger(__name__)
FORMAT = "[ %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

__all__ = [
    "min_max_method",
    "spp_method",
    "cff_method",
    "fft_method",
    "cut_gaussian",
    "ifft_method",
    "args_comp",
    "gaussian_window",
]


def is_inside(value, array):
    try:
        return np.any((value < np.max(array)) & (value > np.min(array)))
    except ValueError:
        return False


def _split_on_SPP(a: np.ndarray, val: Union[List, np.ndarray]) -> List[np.ndarray]:
    """
    Split up an array based on value(s).
    """
    if isinstance(val, numbers.Number):
        if is_inside(val, a):
            v, _ = find_nearest(a, val)
            logger.info(f"split value was set to {v} instead of {val}.")
        else:
            logger.info(
                f"{val} is outside of array range, skipping."
            )
            return [a]
        idx = np.where(a != v)[0]
        return np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    elif isinstance(val, (list, np.ndarray)):
        real_callbacks = []
        for i, v in enumerate(val):
            if not np.any(a == v):
                if is_inside(v, a):
                    value, _ = find_nearest(a, v)
                    real_callbacks.append(value)
                    logger.info(f"{v} was replaced with {value}.")
                else:
                    logger.info(f"{v} was thrown away, not in range..")
            else:
                real_callbacks.append(v)

        idx = np.in1d(a, real_callbacks)
        split_at = a.searchsorted(real_callbacks) - np.arange(0, np.count_nonzero(idx))
        return np.split(a[~idx], split_at)


def _build_single_phase_data(x: np.ndarray, SPP_callbacks: NumericLike = None, flip=False) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array([])
    lastval = 0

    if SPP_callbacks is not None:
        x = _split_on_SPP(x, SPP_callbacks)
    else:
        x = (x,)
    if flip:
        x.insert(0, [])
    logger.info(f"x was split to {len(x)} pieces (including the flip).")
    for index, i in enumerate(x):
        arr = np.asarray(i)
        if index % 2 == 0:
            y = np.append(y, lastval + np.pi * np.arange(1, len(arr) + 1))
        elif index % 2 == 1:
            y = np.append(y, lastval - np.pi * np.arange(1, len(arr) + 1))
        try:
            lastval = y[-1]
        except IndexError:
            lastval = 0

    return np.concatenate(x), y


def min_max_method(
        x,
        y,
        ref,
        sam,
        ref_point,
        maxx=None,
        minx=None,
        SPP_callbacks=None,
):

    x, y = _handle_input(x, y, ref, sam)
    if maxx is None:
        max_ind = argrelextrema(y, np.greater)
        maxx = x[max_ind]
    if minx is None:
        min_ind = argrelextrema(y, np.less)
        minx = x[min_ind]

    _, ref_index = find_nearest(x, ref_point)
    ref_point = x[ref_index]
    logger.info(f"refpoint set to {x[ref_index]} instead of {ref_point}.")

    # subtract the reference point from x axis at extremals
    max_freq = x[ref_index] - maxx
    min_freq = x[ref_index] - minx

    if SPP_callbacks is not None:
        if isinstance(SPP_callbacks, numbers.Number):
            SPP_callbacks -= ref_point
        elif isinstance(SPP_callbacks, (list, np.ndarray)):
            try:
                SPP_callbacks = np.asarray(SPP_callbacks) - ref_point
            except TypeError:
                pass
        else:
            raise TypeError("SPP_callbacks must be list-like, or number.")
        logger.info(f"SPP_callbacks are now {SPP_callbacks}, with ref_point {ref_point}.")

    # find which extremal point is where (relative to reference_point) and order them
    # as they need to be multiplied together with the corresponding order `m`
    neg_freq = np.sort(np.append(max_freq[max_freq < 0], min_freq[min_freq < 0]))[::-1]
    pos_freq = np.sort(np.append(max_freq[max_freq >= 0], min_freq[min_freq >= 0]))

    pos_data_x, pos_data_y = _build_single_phase_data(-pos_freq, SPP_callbacks=SPP_callbacks)

    # if we fail, the whole negative half is empty
    try:
        if np.diff(pos_data_y)[-1] < 0:
            flip = True
            logger.info("Positive side was flipped because the other side is decreasing.")
        else:
            flip = False
    except IndexError:
        flip = False

    neq_data_x, neq_data_y = _build_single_phase_data(-neg_freq, SPP_callbacks=SPP_callbacks, flip=flip)

    x_s = np.insert(neq_data_x, np.searchsorted(neq_data_x, pos_data_x), pos_data_x)
    y_s = np.insert(neq_data_y, np.searchsorted(neq_data_x, pos_data_x), pos_data_y)

    return x_s + ref_point, -y_s + ref_point


def spp_method(delays, omegas, ref_point=0, fit_order=4):
    """
    Calculates the dispersion from SPP's positions and delays.

    Parameters
    ----------

    delays: array-like
    The delay values in fs exactly where omegas taken.

    omegas: array-like
    The angular frequency values where SPP's are located

    ref_point: float
    The reference point in dataset for fitting.

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
        raise ValueError(f"data shapes are different: {delays.shape} & {omegas.shape}")

    idx = np.argsort(omegas)
    omegas, delays = omegas[idx], delays[idx]
    omegas -= ref_point

    _function = _fit_config[fit_order]

    if _has_lmfit:
        fitmodel = Model(_function)
        pars = fitmodel.make_params(**{f"b{i}": 1 for i in range(fit_order + 1)})
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
    return omegas, delays, -dispersion, dispersion_std, bf


def cff_method(x, y, ref, sam, ref_point=0, p0=[1, 1, 1, 1, 1, 1, 1, 1], maxtries=8000):
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

    win_order: int
    The order of gaussian window. Must be even.

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
        pars = fitmodel.make_params(**{f"b{i}": 1 for i in range(fit_order + 1)})
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
        fit_report = "To display detailed results," " you must have `lmfit` installed."
    if show_graph:
        try:
            plot_phase(
                x,
                y,
                result.best_fit,
                bf_fallback=result.best_fit,
                window_title="Phase",
            )
        except UnboundLocalError:

            plot_phase(
                x,
                y,
                _function(x, *popt),
                bf_fallback=_function(x, *popt),
                window_title="Phase",
            )

    return -dispersion, dispersion_std, fit_report
