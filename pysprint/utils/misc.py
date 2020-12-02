import os

import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as st

from pysprint.config import _get_config_value

__all__ = [
    "_unpack_lmfit",
    "find_nearest",
    "_handle_input",
    "_fourier_interpolate",
    "_between",
    "_get_closest",
    "run_from_ipython",
    "_calc_envelope",
    "measurement",
    "_maybe_increase_before_cwt",
    "pad_with_trailing_zeros",
    "pprint_math_or_default",
]


def pprint_math_or_default(s):
    try:
        from IPython.display import display, Math
        display(
            Math(
                s
            )
        )
    except ImportError:
        print(s)


def _maybe_increase_before_cwt(y, tolerance=0.05):
    y = np.asarray(y)
    value = y.min()
    if value - tolerance >= 0:
        return False
    if value < 0 < np.abs(value) - tolerance:
        return True
    return True


def _calc_envelope(x, ind, mode="u"):
    """
    https://stackoverflow.com/a/39662343/11751294
    """
    x_abs = np.abs(x)
    if mode == "u":
        loc = np.where(np.diff(np.sign(np.diff(x_abs))) < 0)[0] + 1
    elif mode == "l":
        loc = np.where(np.diff(np.sign(np.diff(x_abs))) > 0)[0] + 1
    else:
        raise ValueError("mode must be u or l.")
    peak = x_abs[loc]
    envelope = np.interp(ind, loc, peak)
    return envelope, peak, loc


def run_from_ipython():
    """
    Detect if code is run inside Jupyter or maybe Spyder.
    """
    try:
        __IPYTHON__  # noqa
        if any("SPYDER" in name for name in os.environ):
            return False
        return True
    except NameError:
        return False


def _get_closest(x_val, y_val, x_array, y_array):
    """
    Get the closest 2D point in array.
    """
    idx = np.argmin((np.hypot(x_array - x_val, y_array - y_val)))
    value = x_array[idx]
    return value, y_array[idx], idx


def _between(val, except_around):
    if except_around is None:
        return False
    elif len(except_around) != 2:
        raise ValueError(
            f"Invalid interval. Try [start, end] instead of {except_around}"
        )
    else:
        lower = float(min(except_around))
        upper = float(max(except_around))
    if upper >= val >= lower:
        return True
    return False


def _unpack_lmfit(r):
    dispersion, dispersion_std = [], []
    for _, par in r:
        dispersion.append(par.value)
        dispersion_std.append(par.stderr)
    return dispersion, dispersion_std


def measurement(array, confidence=0.95, silent=False):
    """
    Give the measurement results with condifence interval
    assuming the standard deviation is unknown.

    Parameters
    ----------

    array : ndarray
        The array containing the measured values

    confidence : float, optional
        The desired confidence level. Must be _between 0 and 1.

    silent : bool, optional
        Whether to print results immediately. Default is `False`.

    Returns
    -------

    mean: float
        The mean of the given array

    conf: tuple-like (interval)
        The confidence interval

    Examples
    ---------
    >>> import numpy as np
    >>> from pysprint.utils import measurement
    >>> a = np.array([123.783, 121.846, 122.248, 125.139, 122.569])
    >>> mean, interval = measurement(a, 0.99)
    123.117000 ± 2.763022
    >>> mean
    123.117
    >>> interval
    (120.35397798230359, 125.88002201769642)

    Note
    ----
    The results are printed immediately, because people often don't use
    it for further code. Of course, they are also returned if needed.
    """
    precision = _get_config_value("precision")
    mean = np.mean(array)
    conf = st.t.interval(confidence, len(array) - 1, loc=mean, scale=st.sem(array))
    if not silent:
        pprint_math_or_default(
            f"{mean:.{precision}f} ± {(mean - conf[0]):.{precision}f}"
        )
    return mean, conf


def find_nearest(array, value):
    """
    Find the nearest element in array to value.

    Parameters
    ----------
    array : np.ndarray-like
        The array to search in.
    value : float
        The value to search.

    Returns
    -------
    value : float
        The closest value in array.
    idx : int
        The index of the closest element.
    """
    array = np.asarray(array)
    idx = (np.abs(value - array)).argmin()
    return array[idx], idx


def _handle_input(x, y, ref, sam):
    """
    Instead of handling the inputs in every function, there is this private method.

    Parameters
    ----------

    x: array-like
    x-axis data

    y: array-like
    y-axis data

    ref, sam: array-like
    reference and sample arm spectrum evaluated at x

    Returns
    -------
    x: array-like
    unchanged x data

    Ydata: array-like
    the transformed y data

    """
    if len(x) == 0:
        raise ValueError("No values for x.")

    if len(y) == 0:
        raise ValueError("No values for y.")

    if (len(x) > 0) and (len(ref) > 0) and (len(sam) > 0):
        y_data = (y - ref - sam) / (2 * np.sqrt(ref * sam))
    elif (len(ref) == 0) or (len(sam) == 0):
        y_data = y
    else:
        raise TypeError("Input types are wrong.\n")
    return np.asarray(x), np.asarray(y_data)


def _fourier_interpolate(x, y):
    """ Simple linear interpolation for FFTs"""
    xs = np.linspace(x[0], x[-1], len(x))
    intp = interp1d(x, y, kind="linear", fill_value="extrapolate")
    ys = intp(xs)
    return xs, ys


def pad_with_trailing_zeros(array, shape):
    """
    Pad an array with trailing zeros to be the desired shape
    """
    c = shape - len(array)
    if c < 1:
        return np.asarray(array)
    return np.pad(array, pad_width=(0, c), mode="constant")
