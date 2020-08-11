"""
Methods for manipulating the loaded data
"""
import numpy as np
from scipy.signal import (
    find_peaks,
    savgol_filter,
    gaussian,
    convolve,
    find_peaks_cwt,
)
from scipy.interpolate import interp1d
from pysprint.utils import (
    find_nearest,
    _handle_input,
    between,
    _maybe_increase_before_cwt,
)

# TODO : all of these methods should be rewritten, they are poor quality.


def cwt(x, y, ref, sam, widths, floor_thres=0.1):
    x, y = _handle_input(x, y, ref, sam)
    idx = find_peaks_cwt(y, widths=widths)
    if _maybe_increase_before_cwt(y, tolerance=floor_thres):
        y += 2
    y_rec = 1 / y
    idx2 = find_peaks_cwt(y_rec, widths=widths)
    return x[idx], y[idx] - 2, x[idx2], y[idx2] - 2


def savgol(x, y, ref, sam, window=101, order=3):
    x, y = _handle_input(x, y, ref, sam)
    xint, yint = interpolate_data(x, y, [], [])
    if window > order:
        try:
            if window % 2 == 1:
                fil = savgol_filter(yint, window_length=window, polyorder=order)
                return xint, fil
            else:
                fil = savgol_filter(yint, window_length=window + 1, polyorder=order)
                return xint, fil
        except Exception as e:
            print(e)
    else:
        raise ValueError("Order must be lower than window length.")


def find_peak(x, y, ref, sam, pro_max=1, pro_min=1, threshold=0.1, except_around=None):
    if except_around is not None and len(except_around) != 2:
        raise ValueError("Invalid except_around arg. Try [start, stop].")
    if except_around is not None:
        try:
            float(except_around[0])
            float(except_around[1])
        except ValueError:
            raise ValueError(
                "Invalid except_around arg. Only numeric values are allowed."
            )
    x, y = _handle_input(x, y, ref, sam)
    max_indexes, _ = find_peaks(y, prominence=pro_max)
    y_rec = 1 / y
    min_indexes, _ = find_peaks(y_rec, prominence=pro_min)
    min_idx = []
    max_idx = []
    for idx in max_indexes:
        if between(x[idx], except_around) or np.abs(y[idx]) > threshold:
            max_idx.append(idx)
    for idx in min_indexes:
        if between(x[idx], except_around) or np.abs(y[idx]) > threshold:
            min_idx.append(idx)

    if len(x[max_idx]) != len(y[max_idx]) or len(x[min_idx]) != len(y[min_idx]):
        raise ValueError("Something went wrong, try to cut the edges of data.")

    return x[max_idx], y[max_idx], x[min_idx], y[min_idx]


def convolution(x, y, ref, sam, win_len, standev=200):
    x, y = _handle_input(x, y, ref, sam)
    if win_len < 0 or win_len > len(x):
        raise ValueError("Window length must be 0 < window_length < len(x)")

    xint, yint = interpolate_data(x, y, [], [])
    window = gaussian(win_len, std=standev)
    smoothed = convolve(yint, window / window.sum(), mode="same")
    return xint, smoothed


def interpolate_data(x, y, ref, sam):
    x, y = _handle_input(x, y, ref, sam)
    xint = np.linspace(x[0], x[-1], len(x))
    intp = interp1d(x, y, kind="linear")
    yint = intp(xint)
    return xint, yint


def cut_data(x, y, ref, sam, start=None, stop=None):
    x, y = _handle_input(x, y, ref, sam)
    if start is None:
        start = np.min(x)
    if stop is None:
        stop = np.max(x)
    if start < stop:
        low_item, _ = find_nearest(x, start)
        high_item, _ = find_nearest(x, stop)
        mask = np.where((x >= low_item) & (x <= high_item))
        return x[mask], y[mask]
    elif stop < start:
        raise ValueError("Start must not exceed stop value.")
    else:
        return np.array([]), np.array([])
