"""
This file contains code from a blog post by Jake VanderPlas.

See at:
https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

References:

[1] Dutt A., Rokhlin V. : Fast Fourier Transforms for Nonequispaced Data II,
    Applied and Computational Harmonic Analysis
    Volume 2, Issue 1, January 1995, Pages 85-100
    (1995)

[2] Greengard, Leslie & Lee, June-Yub.: Accelerating the
    Nonuniform Fast Fourier Transform,
    Society for Industrial and Applied Mathematics.
    46. 443-454. 10.1137/S003614450343200X.
    (2004)
"""

import warnings
from inspect import isfunction

import numpy as np

try:
    from numba import jit

    _has_numba = True
except ImportError:
    _has_numba = False

    def jit(func=None, *args, **kwargs):
        """Replace jit if numba is not available."""

        def _jit(f):
            return f

        if isfunction(func):
            return func
        else:
            return _jit


def _grid_params(gl, eps):
    if eps <= 1e-33 or eps >= 1e-1:
        raise ValueError(f"eps = {eps:.0e}, but it must satisfy " "1e-33 < eps < 1e-1.")

    ratio = 2 if eps > 1e-11 else 3
    Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
    Mr = max(ratio * gl, 2 * Msp)
    lambda_ = Msp / (ratio * (ratio - 0.5))
    tau = np.pi * lambda_ / gl ** 2
    return Mr, Msp, tau


@jit(nopython=True)
def _grid(x, c, tau, Msp, ftau, E3):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr

    for j in range(Msp + 1):
        E3[j] = np.exp(-((np.pi * j / Mr) ** 2) / tau)

    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        xi = xi - hx * m
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (Mr * tau))
        E2mm = 1
        for mm in range(Msp):
            ftau[(m + mm) % Mr] += c[i] * E1 * E2mm * E3[mm]
            E2mm *= E2
            ftau[(m - mm - 1) % Mr] += c[i] * E1 / E2mm * E3[mm + 1]


def _compute_gaussian_grid(x, c, Mr, Msp, tau):
    ftau = np.zeros(Mr, dtype=c.dtype)
    E3 = np.zeros(Msp + 1, dtype=x.dtype)
    _grid(x, c, tau, Msp, ftau, E3)
    return ftau


def nuifft_times(gl, df=1):
    """Compute the time range used in nufft for gl time bins"""
    return df * np.arange(-(gl // 2), gl - (gl // 2))


def _compute_gaussian_grid_nonumba(x, c, Mr, Msp, tau):
    """Compute the 1D gaussian gridding with Numpy"""
    N = len(x)
    ftau = np.zeros(Mr, dtype=c.dtype)
    hx = 2 * np.pi / Mr
    xmod = x % (2 * np.pi)

    m = 1 + (xmod // hx).astype(int)
    msp = np.arange(-Msp, Msp)[:, np.newaxis]
    mm = m + msp

    E1 = np.exp(-0.25 * (xmod - hx * m) ** 2 / tau)

    # Basically in the following lines we compute
    # this in a tricky way:
    # E2 = np.exp(msp * (xmod - hx * m) * np.pi / (Mr * tau))

    E2 = np.empty((2 * Msp, N), dtype=xmod.dtype)
    E2[Msp] = 1
    E2[Msp + 1:] = np.exp((xmod - hx * m) * np.pi / (Mr * tau))
    E2[Msp + 1:].cumprod(0, out=E2[Msp + 1:])
    E2[Msp - 1::-1] = 1.0 / (E2[Msp + 1] * E2[Msp:])

    E3 = np.exp(-((np.pi * msp / Mr) ** 2) / tau)
    spread = (c * E1) * E2 * E3

    np.add.at(ftau, mm % Mr, spread)

    return ftau


def nuifft(x, y, gl, df=1.0, epsilon=1e-12, exponent="positive"):
    """
    Non-Uniform (inverse) Fast Fourier Transform to avoid linear
    interpolation of interferograms.

    Compute the non-uniform FFT of one-dimensional points x with (complex)
    values y. Result is computed at frequencies (df * gl)
    for integer gl in the range -gl/2 < m < gl/2.
    Uses the fast Gaussian grid algorithm of Greengard & Lee (2004)

    Parameters
    ----------
    x, y : array-like
        real locations x and (complex) values y to be transformed.
    gl, df : int & float
        Parameters specifying the desired frequency grid. Transform will be
        computed at frequencies df * (-(gl//2) + arange(gl))
    epsilon : float
        The desired approximate error for the FFT result. Must be in range
        1E-33 < eps < 1E-1, though be aware that the errors are only well
        calibrated near the range 1E-12 ~ 1E-6.
    exponent : str
        if 'negative', compute the transform with a negative exponent.
        if 'positive', compute the transform with a positive exponent.

    Returns
    -------
    F_k : ndarray
        The complex discrete Fourier transform

    Notes
    -----
    If numba is not installed it's approximately 5x times slower.
    """
    if exponent not in ("positive", "negative"):
        raise ValueError("exponent must be `positive` or `negative`.")

    x = df * np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError("Expected one-dimensional input arrays")

    if x.shape != y.shape:
        raise ValueError("Array shapes must match")

    gl = int(gl)
    N = len(x)
    k = nuifft_times(gl)

    Mr, Msp, tau = _grid_params(gl, epsilon)

    if _has_numba:
        ftau = _compute_gaussian_grid(x, y, Mr, Msp, tau)
    else:
        warnings.warn("Numba is not available, falling back to slower version.")
        ftau = _compute_gaussian_grid_nonumba(x, y, Mr, Msp, tau)

    if exponent == "negative":
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(gl // 2):], Ftau[:gl // 2 + gl % 2]])

    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau
