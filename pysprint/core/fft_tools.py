import warnings

import numpy as np
from scipy.signal import find_peaks


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def find_roi(x, y):
    y = np.abs(y)
    idx = np.where(x > 0)
    return x[idx], y[idx]


def find_center(x, y, n_largest=5, return_multiple=None):

    peaks, props = find_peaks(y, prominence=0.001, height=np.max(y) / 100)

    if n_largest > len(props["prominences"]):
        n_largest = len(props["prominences"])

    # find the most outlying peaks from the noise
    ind1 = np.argpartition(props["prominences"], -n_largest)[-n_largest:]
    # find the highest peaks by value
    ind2 = np.argpartition(props["peak_heights"], -n_largest)[-n_largest:]

    ind = np.unique(np.concatenate((ind1, ind2)))

    peaks = peaks[ind]
    y_prob_density = np.exp((-((x - np.max(x) / 2.5) ** 2)) / (1000 * np.max(x)))
    _x, _y, _y_ = x[peaks], y_prob_density[peaks], y[peaks]
    # weighted with the prob density function above from origin
    try:
        residx = np.argmax(_x * _y * _y_)
    except ValueError as err:
        msg = ValueError(
            "Probably you need to set bigger window FWHM. "
            "After IFFT, the center of the peak could not be determined."
        )
        raise msg from err

    if return_multiple:
        try:
            N = int(return_multiple)
        except ValueError as err:
            raise ValueError("Must return integer number of peaks.") from err
        xval, yval = x[peaks], y[peaks]
        if len(xval) < N:
            N = len(xval)
        idx_ = np.argsort(xval)
        retx, rety = xval[idx_], yval[idx_]
        return retx[:N], rety[:N]

    return _x[residx], _y_[residx]


# THIS FUNCTION IS EXPERIMENTAL!
# def __find_center(x, y, n_largest=5, return_multiple=None):
#     if return_multiple:
#         try:
#             N = int(return_multiple)
#         except ValueError as err:
#             raise ValueError("Must return integer number of peaks.") from err
#
#     dist = 150 if len(x) < 5000 else len(x) / 50
#     peaks, props = find_peaks(y, prominence=0.001, height=np.max(y) / 100, distance=dist)
#
#     if n_largest > len(props["prominences"]):
#         n_largest = len(props["prominences"])
#
#     # find the most outlying peaks from the noise
#     ind1 = np.argpartition(props["prominences"], -n_largest)[-n_largest:]
#     # find the highest peaks by value
#     ind2 = np.argpartition(props["peak_heights"], -n_largest)[-n_largest:]
#
#     candidates = np.intersect1d(ind1, ind2)
#     ind = np.unique(np.concatenate((ind1, ind2)))
#
#     if return_multiple:
#
#         if N <= len(ind2):
#             pks = peaks[ind1][:N]
#             return x[pks], y[pks]
#
#         elif N == len(candidates):
#             pks = peaks[candidates]
#             return x[pks], y[pks]
#
#         elif N > len(ind2) and N > len(candidates):
#             _peaks, _props = find_peaks(y, prominence=0.001, height=np.max(y) / 100, distance=dist / 2)
#             try:
#                 _ind2 = np.argpartition(_props["peak_heights"], -N)[-N:]
#             except ValueError as err:
#                 msg = ValueError("Not enough peaks found..")
#                 raise msg from err
#
#             pks = _peaks[_ind2]
#             return x[pks], y[pks]
#         else:
#             raise ValueError("Not enough peaks found..")
#
#     peaks = peaks[ind]
#
#     y_prob_density = np.exp((-((x - np.max(x) / 2.5) ** 2)) / (1000 * np.max(x)))
#     _x, _y, _y_ = x[peaks], y_prob_density[peaks], y[peaks]
#     # weighted with the prob density function above from origin
#     try:
#         residx = np.argmax(_x * _y * _y_)
#     except ValueError as err:
#         msg = ValueError(
#             "Probably you need to set bigger window FWHM. "
#             "After IFFT, the center of the peak could not be determined."
#         )
#         raise msg from err
#
#     return _x[residx], _y_[residx]


def _ensure_window_at_origin(center, fwhm, order, peak_center_height, tol=1e-3):
    """
    Ensure that the gaussian window of given parameters is
    not crossing zero with bigger value than the desired tolerance.
    """
    std = fwhm / (2 * (np.log(2) ** (1 / order)))
    val = np.exp(-((0 - center) ** order) / (std ** order))
    return val < peak_center_height * tol, val


def predict_fwhm(x, y, center, peak_center_height, prefer_high_order=True, tol=1e-3):
    if np.iscomplexobj(y):
        y = np.abs(y)

    N = len(x)

    peak_position = center / N

    # explicitly prefer higher order windows
    # if the peak is too close to the origin

    if peak_position < 0.2:
        warnings.warn(
            "The peak is too close to the origin, manual control is advised.",
            UserWarning,
        )
        prefer_high_order = True

    if prefer_high_order:
        order_choices = (8, 10, 12)
    else:
        order_choices = (2, 4, 6)

    sgnl = signaltonoise(y)
    if peak_position < 0.6:
        if sgnl > 0.1:
            center += N * 0.05 + (sgnl - 0.1) * N * 2
            window_size = 0.75 * center * 2 + (sgnl - 0.1) * N * 0.2
        else:
            center += N * 0.05
            window_size = 0.75 * center * 2
    else:
        if sgnl > 0.1:
            center += N * 0.05 + (sgnl - 0.1) * N * 2
            window_size = 0.5 * center * 2 + (sgnl - 0.1) * N * 0.2
        else:
            center += N * 0.05
            window_size = 0.5 * center * 2
    try:
        order = order_choices[int(2 - (3 * peak_position // 2))]
    except IndexError:
        order = order_choices[0]

    if _ensure_window_at_origin(
            center, window_size, order, peak_center_height, tol=tol
    )[0]:
        return center, window_size, order
    else:
        val = _ensure_window_at_origin(
            center, window_size, order, peak_center_height, tol=tol
        )[1]
        warnings.warn(
            "The window is bigger at the origin than the desired tolerance. "
            f"Actual:{val:.4e}, Desired:{tol * peak_center_height:.4e}"
        )
        return center, window_size, order + 2


def _run(
        ifg, skip_domain_check=False, show_graph=True, usenifft=False,
):
    print("Interferogram received.")
    if ifg.probably_wavelength is True and not skip_domain_check:
        print(
            "Probably in wavelength domain, changing to frequency...",
            end="",
            flush=True,
        )
        ifg.chdomain()
        print("Done")
    if usenifft:
        pprint = " using NUIFFT algorithm..."
    else:
        pprint = "..."
    print(f"Applying IFFT{pprint}", end="", flush=True)
    ifg.ifft(usenifft=usenifft)
    print("Done")
    x, y = find_roi(ifg.x, ifg.y)
    a, b = find_center(x, y)
    print("Acquiring gaussian window parameters...", end="", flush=True)
    print("Done")
    c, ws, _order = predict_fwhm(x, y, center=a, peak_center_height=b)
    print(
        f"A {_order} order gaussian window centered at "
        f"{c:.2f} fs with FWHM {ws:.2f} fs was applied."
    )
    ifg.window(at=c, fwhm=ws, window_order=_order, plot=show_graph)
    ifg.apply_window()
    print("Applying FFT...", end="", flush=True)
    ifg.fft()
    print("Done")
    print("Calculating...")
