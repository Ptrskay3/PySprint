# -*- coding: utf-8 -*-
from math import factorial
import operator

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

try:
    from lmfit import Model
    _has_lmfit = True
except ImportError:
    _has_lmfit = False


from pysprint.utils import (
    find_nearest, _handle_input, lmfit_disp, fourier_interpolate
    )


__all__ = [
    'min_max_method', 'cos_fit1', 'cos_fit2', 'cos_fit3',
    'cos_fit4', 'cos_fit5', 'spp_method', 'cff_method', 'fft_method',
    'cut_gaussian', 'ifft_method', 'args_comp'
    ]


_fit_config = { 
    1: poly1,  # noqa
    2: poly2,  # noqa
    3: poly3,  # noqa
    4: poly4,  # noqa
    5: poly5  # noqa
}


def min_max_method(
        x, y, ref, sam, ref_point,
        maxx=None, minx=None, fit_order=5, show_graph=False
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
    if True returns a matplotlib plot and pauses execution until closing the window

    Returns
    -------

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    dispersion_std: array-like
    [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

    fit_report: lmfit report
    """
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

    neg_freq = np.sort(np.append(max_freq[max_freq < 0], min_freq[min_freq < 0]))[::-1]
    pos_freq = np.sort(np.append(max_freq[max_freq > 0], min_freq[min_freq > 0]))

    if len(neg_freq) == 0 and len(pos_freq) == 0:
        raise ValueError('No extremal points found.')

    pos_values = np.pi * np.arange(1, len(pos_freq) + 1)
    neg_values = np.pi * np.arange(1, len(neg_freq) + 1)

    x_s = np.append(pos_freq, neg_freq)
    y_s = np.append(pos_values, neg_values)

    # FIXME: Do we even need this? 
    # Yes, we do. This generates a prettier plot.
    idx = np.argsort(x_s)
    full_x, full_y = x_s[idx], y_s[idx]
    
    if fit_order not in range(6):
        raise ValueError('fit order must be in [1, 5]')

    _function = _fit_config[fit_order]

    if _has_lmfit:
        fitmodel = Model(_function)
        pars = fitmodel.make_params(**{f'b{i}':1 for i in range(fit_order + 1)})
        result = fitmodel.fit(full_y, x=full_x, params=pars)
    else:
        popt, pcov = curve_fit(_function, full_x, full_y, maxfev=8000)

    try:
        if _has_lmfit:
            dispersion, dispersion_std = lmfit_disp(result.params.items())
            dispersion = dispersion[1:]
            dispersion_std = dispersion_std[1:]
            for idx in range(len(dispersion)):
                dispersion[idx] = dispersion[idx] * factorial(idx+1)
                dispersion_std[idx] = dispersion_std[idx] * factorial(idx+1)
            while len(dispersion) < 5:
                dispersion.append(0)
                dispersion_std.append(0)
            fit_report = result.fit_report()
        else:
            full_x = np.asarray(full_x)
            dispersion = []
            dispersion_std = []
            for idx in range(len(popt) - 1):
                dispersion.append(popt[idx + 1] * factorial(idx + 1))
            while len(dispersion) < 5:
                dispersion.append(0)
            while len(dispersion_std) < len(dispersion):
                dispersion_std.append(0)
            fit_report = 'To display detailed results, you must have lmfit installed.'
        if show_graph:
            fig = plt.figure(figsize=(7, 7))
            fig.canvas.set_window_title('Min-max method fitted')
            plt.plot(full_x, full_y, 'o', label='dataset')
            try:
                plt.plot(full_x, result.best_fit, 'r*', label='fitted')
            except Exception:
                plt.plot(full_x, _function(full_x, *popt), 'r*', label='fitted')
            plt.legend()
            plt.grid()
            plt.show()
        return dispersion, dispersion_std, fit_report
    except Exception as e:
        return [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], e

# TODO: implement higher orders

def poly6(x, b0, b1, b2, b3, b4, b5, b6):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6
    b4 = FOD / 24
    b5 = QOD / 120
    b6 = SOD / 720
    """
    return b0+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5+b6*x**6

def poly5(x, b0, b1, b2, b3, b4, b5):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6
    b4 = FOD / 24
    b5 = QOD / 120
    """
    return b0+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5


def poly4(x, b0, b1, b2, b3, b4):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6
    b4 = FOD / 24
    """
    return b0+b1*x+b2*x**2+b3*x**3+b4*x**4


def poly3(x, b0, b1, b2, b3):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6

    """
    return b0+b1*x+b2*x**2+b3*x**3


def poly2(x, b0, b1, b2):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    """
    return b0+b1*x+b2*x**2


def poly1(x, b0, b1):
    """
    Taylor polynomial for fit
    b1 = GD
    """
    return b0+b1*x


def cos_fit1(x, c0, c1, b0, b1):
    return c0 + c1 * np.cos(poly1(x, b0, b1))


def cos_fit2(x, c0, c1, b0, b1, b2):
    return c0 + c1 * np.cos(poly2(x, b0, b1, b2))


def cos_fit3(x, c0, c1, b0, b1, b2, b3):
    return c0 + c1 * np.cos(poly3(x, b0, b1, b2, b3))


def cos_fit4(x, c0, c1, b0, b1, b2, b3, b4):
    return c0 + c1 * np.cos(poly4(x, b0, b1, b2, b3, b4))


def cos_fit5(x, c0, c1, b0, b1, b2, b3, b4, b5):
    return c0 + c1 * np.cos(poly5(x, b0, b1, b2, b3, b4, b5))

# TODO: implement higher order
def cos_fit6(x, c0, c1, b0, b1, b2, b3, b4, b5, b6):
    return c0 + c1 * np.cos(poly6(x, b0, b1, b2, b3, b4, b5, b6))


def spp_method(delays, omegas, ref_point=0, fit_order=4, from_raw=False):
    """
    Calculates the dispersion from SPP's positions and delays.

    Parameters
    ----------

    delays: array-like
    the time delays in fs
    if from_raw is enabled you must pass matching pairs with omegas

    omegas: array-like
    in form of [[SPP1, SPP2, SPP3, SPP4],[SPP1, SPP2, SPP3, SPP4], ..]
    for lesser SPP cases replace elements with None:
    [[SPP1, None, None, None],[SPP1, None, None, None], ..]
    if from_raw is enabled, you must pass matching pairs with delays

    fit_order: int
    order of polynomial to fit the given data

    from_raw: bool
    if True you can pass matching pairs to delays and omegas, and it will perform
    a normal curve fitting. It's useful at the API.

    Returns
    -------
    omegas_unpacked: array-like
    x axis data

    delays_unpacked : array-like
    y axis data

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    dispersion_std: array-like
    [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

    bf: array-like
    best fitting curve for plotting
    """
    if from_raw:
        delays_unpacked = delays
        omegas_unpacked = omegas
    else:
        delays = delays[delays != np.array(None)]
        omegas_unpacked = []
        delays_unpacked = []
        for delay, element in zip(delays, omegas):
            item = [x for x in element if x is not None]
            omegas_unpacked.extend(item)
            delays_unpacked.extend(len(item) * [delay])
    # FIXME: should be numpy arrays..
    L = sorted(zip(omegas_unpacked, delays_unpacked), key=operator.itemgetter(0))
    omegas_unpacked, delays_unpacked = zip(*L)
    omegas_unpacked = [val-ref_point for val in omegas_unpacked]
    try:
        if _has_lmfit:
            if fit_order == 2:
                fitModel = Model(poly2)
                params = fitModel.make_params(b0=1, b1=1, b2=1)
                result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params=params, method='leastsq')
            elif fit_order == 3:
                fitModel = Model(poly3)
                params = fitModel.make_params(b0=1, b1=1, b2=1, b3=1)
                result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params=params, method='leastsq')
            elif fit_order == 4:
                fitModel = Model(poly4)
                params = fitModel.make_params(b0=1, b1=1, b2=1, b3=1, b4=1)
                result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params=params, method='leastsq')
            elif fit_order == 1:
                fitModel = Model(poly1)
                params = fitModel.make_params(b0=1, b1=1)
                result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params=params, method='leastsq')
            else:
                raise ValueError('Order is out of range, please select from [1,4]')
            dispersion, dispersion_std = lmfit_disp(result.params.items())
            for idx in range(len(dispersion)):
                dispersion[idx] = dispersion[idx]*factorial(idx)
            for idx in range(len(dispersion_std)):
                if dispersion_std[idx] is not None:
                    dispersion_std[idx] = dispersion_std[idx] * factorial(idx)
                else:
                    dispersion_std[idx] = 0
            while len(dispersion) < 5:
                dispersion.append(0)
                dispersion_std.append(0)
            while len(dispersion_std) < 5:
                dispersion_std.append(0)
            bf = result.best_fit
        else:
            if fit_order == 4:
                popt, pcov = curve_fit(poly4, omegas_unpacked, delays_unpacked, maxfev=8000)
                _function = poly4
            elif fit_order == 3:
                popt, pcov = curve_fit(poly3, omegas_unpacked, delays_unpacked, maxfev=8000)
                _function = poly3
            elif fit_order == 2:
                popt, pcov = curve_fit(poly2, omegas_unpacked, delays_unpacked, maxfev=8000)
                _function = poly2
            elif fit_order == 1:
                popt, pcov = curve_fit(poly1, omegas_unpacked, delays_unpacked, maxfev=8000)
                _function = poly1
            else:
                raise ValueError('Order is out of range, please select from [1,4]')
            omegas_unpacked = np.asarray(omegas_unpacked)
            dispersion = []
            dispersion_std = []
            for idx in range(len(popt)):
                dispersion.append(popt[idx] * factorial(idx))
            while len(dispersion) < 5:
                dispersion.append(0)
            while len(dispersion_std) < len(dispersion):
                dispersion_std.append(0)
            bf = _function(omegas_unpacked, *popt)
        return omegas_unpacked, delays_unpacked, dispersion, dispersion_std, bf
    except Exception as e:
        return [], [], [e], [], [] # ??? this must be a wrong way of treating errors.


def cff_method(
        x, y, ref, sam, ref_point=0,
        p0=[1, 1, 1, 1, 1, 1, 1, 1], maxtries=8000
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
    # TODO: BOUNDS WILL BE SET  ..
    # bounds=((-1000, -10000, -10000, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),
    # (1000, 10000, 10000, np.inf, np.inf, np.inf, np.inf, np.inf))
    x, y = _handle_input(x, y, ref, sam)

    try:
        if len(np.trim_zeros(p0, 'b')) + 4 == len(p0):
            _funct = cos_fit1
            p0 = p0[:-4]
        elif p0[-1] == 0 and p0[-2] == 0 and p0[-3] == 0:
            _funct = cos_fit2
            p0 = p0[:-3]
        elif p0[-1] == 0 and p0[-2] == 0:
            _funct = cos_fit3
            p0 = p0[:-2]
        elif p0[-1] == 0:
            _funct = cos_fit4
            p0 = p0[:-1]
        else:
            _funct = cos_fit5
        popt, pcov = curve_fit(_funct, x-ref_point, y, p0, maxfev=maxtries)
        dispersion = np.zeros_like(popt)[:-3]
        for num in range(len(popt)-3):
            dispersion[num] = popt[num+3]*factorial(num+1)
        return dispersion, _funct(x-ref_point, *popt)
    except RuntimeError:
        raise ValueError(f'Max tries ({maxtries}) reached.. \nParameters could not be estimated.')


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
    yf = scipy.fftpack.fft(y)
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
    std = fwhm/(2 * (np.log(2)**(1 / order)))
    return np.exp(-((t - tau)**order)/(std**order))


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
    y = y * gaussian_window(x, tau=spike, fwhm=fwhm, order=win_order)
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
    xf = np.fft.fftfreq(N, d=(x[1]-x[0])/(2*np.pi))
    yf = np.fft.ifft(y)
    return xf, yf


def args_comp(x, y, ref_point=0, fit_order=5, show_graph=False):
    """
    Calculates the phase of complex dataset then unwrap by changing deltas between
    values to 2*pi complement. At the end, fit a polynomial curve to determine
    dispersion coeffs.

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
    if True returns a matplotlib plot and pauses execution until closing the window

    Returns
    -------

    dispersion: array-like
    [GD, GDD, TOD, FOD, QOD]

    dispersion_std: array-like
    [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

    fit_report: lmfit report
    """
    if fit_order not in range(6):
        raise ValueError('fit order must be in [1, 5]')

    x -= ref_point
    # shifting to [0, 2pi] if necessary
    # angles = (angles + 2 * np.pi) % (2 * np.pi)
    y = np.unwrap(np.angle(y), axis=0)

    _function = _fit_config[fit_order]

    fitmodel = Model(_function)
    pars = fitmodel.make_params(**{f'b{i}':1 for i in range(fit_order + 1)})
    result = fitmodel.fit(y, x=x, params=pars)

    try:
        dispersion, dispersion_std = lmfit_disp(result.params.items())
        dispersion = dispersion[1:]
        dispersion_std = dispersion_std[1:]
        for idx in range(len(dispersion)):
            dispersion[idx] = dispersion[idx] * factorial(idx+1)
            dispersion_std[idx] = dispersion_std[idx] * factorial(idx+1)
        while len(dispersion) < 5:
            dispersion.append(0)
            dispersion_std.append(0)
        fit_report = result.fit_report()
        if show_graph:
            fig = plt.figure(figsize=(7, 7))
            fig.canvas.set_window_title('Phase')
            plt.plot(x, y, 'o', label='dataset')
            plt.plot(x, result.best_fit, 'r--', label='fitted')
            plt.legend()
            plt.grid()
            plt.show()
        return dispersion, dispersion_std, fit_report
    except Exception as e:
        return [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], e
