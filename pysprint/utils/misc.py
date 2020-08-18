import os
import sys
import re
from copy import copy
from functools import wraps, lru_cache
import threading
import time
from itertools import cycle

import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as st

__all__ = [
    "unpack_lmfit",
    "find_nearest",
    "_handle_input",
    "print_disp",
    "fourier_interpolate",
    "between",
    "get_closest",
    "run_from_ipython",
    "calc_envelope",
    "measurement",
    "_maybe_increase_before_cwt",
    "pad_with_trailing_zeros",
    "mutually_exclusive_args",
    "lazy_property",
    "inplacify",
    "progress"
]


def progress(func):
    active = threading.Lock()

    def spinning_pbar_printer():
        symbols = ['|', '/', '-', '\\', '\\']
        cursor = cycle(symbols)
        while active.locked():
            sys.stdout.write("\r")
            sys.stdout.write("Working... " + next(cursor))
            sys.stdout.flush()
            time.sleep(0.1)

    def wrapper(*args, **kwargs):
        t = threading.Thread(target=spinning_pbar_printer)
        active.acquire()
        t.start()
        try:
            res = func(*args, **kwargs)
        finally:
            active.release()
        return res

    return wrapper


_inplace_doc = """\n\tinplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.\n\n\t"""


def _has_parameter_section(method):
    try:
        return "Parameters" in method.__doc__
    except TypeError:
        return False


def update_doc(method, doc):
    if _has_parameter_section(method):
        newdoc = _build_doc(method, doc)
        method.__doc__ = newdoc
    else:
        newdoc = """\n\tParameters
        ----------\n\tinplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.\n\n\t"""

        nodoc_head = (f"Docstring automatically created for {method.__name__}. "
                      "Parameter list may not be complete.\n")
        if method.__doc__ is not None:
            method.__doc__ += newdoc
        else:
            method.__doc__ = nodoc_head + newdoc
        return


def _build_doc(method, param):
    patt = r"(\w+(?=\s*[-]{4,}[^/]))"  # finding sections
    splitted_doc = re.split(patt, method.__doc__)
    try:
        target = splitted_doc.index("Parameters") + 1
    except ValueError:
        return method.__doc__

    splitted_doc[target] = splitted_doc[target].rstrip() + param

    return ''.join(_ for _ in splitted_doc if _ is not None)


def inplacify(method):
    update_doc(method, _inplace_doc)

    @wraps(method)
    def wrapper(self, *args, **kwds):
        inplace = kwds.pop("inplace", True)
        if inplace:
            method(self, *args, **kwds)
        else:
            return method(copy(self), *args, **kwds)

    return wrapper


# https://stackoverflow.com/a/54487188/11751294
def mutually_exclusive_args(keyword, *keywords):
    """"
    Decorator to restrict the user to specify exactly one of the given parameters.
    Often used for std and fwhm for Gaussian windows.
    """
    keywords = (keyword,) + keywords

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if sum(k in keywords for k in kwargs) != 1:
                raise TypeError(
                    "You must specify exactly one of {}.".format(" and ".join(keywords))
                )
            return func(*args, **kwargs)

        return inner

    return wrapper


def lazy_property(f):
    return property(lru_cache()(f))


def _maybe_increase_before_cwt(y, tolerance=0.05):
    y = np.asarray(y)
    value = y.min()
    if value - tolerance >= 0:
        return False
    if value < 0 < np.abs(value) - tolerance:
        return True
    return True


def calc_envelope(x, ind, mode="u"):
    """Source: https://stackoverflow.com/a/39662343/11751294
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
    Detect explicitly if code is run inside Jupyter.
    """
    try:
        __IPYTHON__
        if any("SPYDER" in name for name in os.environ):
            return False
        return True
    except NameError:
        return False


def get_closest(x_val, y_val, x_array, y_array):
    idx = np.argmin((np.hypot(x_array - x_val, y_array - y_val)))
    value = x_array[idx]
    return value, y_array[idx], idx


def between(val, except_around):
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


def unpack_lmfit(r):
    dispersion, dispersion_std = [], []
    for name, par in r:
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
        The desired confidence level. Must be between 0 and 1.

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

    Notes
    -----

    I decided to print the results immediately, because people often don't use
    it for further code. Of course, they are also returned if needed.
    """
    mean = np.mean(array)
    conf = st.t.interval(confidence, len(array) - 1, loc=mean, scale=st.sem(array))
    if not silent:
        print(f"{mean:5f} ± {(mean - conf[0]):5f}")
    return mean, conf


def find_nearest(array, value):
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


def print_disp(f):
    @wraps(f)
    def wrapping(*args, **kwargs):
        disp, disp_std, stri = f(*args, **kwargs)
        labels = ("GD", "GDD", "TOD", "FOD", "QOD", "SOD")
        disp = np.trim_zeros(disp, "b")
        disp_std = disp_std[: len(disp)]
        for i, (label, disp_item, disp_std_item) in enumerate(
                zip(labels, disp, disp_std)
        ):
            if run_from_ipython():
                from IPython.display import display, Math

                display(
                    Math(f"{label} = {disp_item:.5f} ± {disp_std_item:.5f} fs^{i + 1}")
                )
            else:
                print(f"{label} = {disp_item:.5f} ± {disp_std_item:.5f} fs^{i + 1}")
        return disp, disp_std, stri

    return wrapping


def fourier_interpolate(x, y):
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
    return np.pad(array, pad_width=(0, c), mode="constant")
