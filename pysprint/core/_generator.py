"""
Sample generator
"""
import numpy as np

from pysprint.core.bases._dataset_base import C_LIGHT


def _ensure_input(start, stop, center, resolution, pulse_width):
    if start >= stop:
        raise ValueError("start value must be less than stop")
    if center < start or center > stop:
        raise ValueError("center must be _between start and stop")
    if resolution > (stop - start):
        raise ValueError("resolution is too big")
    if pulse_width <= 0:
        raise ValueError("Pulse width is strictly positive.")


def _disp(x, GD=0, GDD=0, TOD=0, FOD=0, QOD=0, SOD=0):
    return (
        x * GD
        + (GDD / 2) * x ** 2
        + (TOD / 6) * x ** 3
        + (FOD / 24) * x ** 4
        + (QOD / 120) * x ** 5
        + (SOD / 720) * x ** 6
    )


def generator_freq(
    start,
    stop,
    center,
    delay,
    GD=0,
    GDD=0,
    TOD=0,
    FOD=0,
    QOD=0,
    SOD=0,
    resolution=0.1,
    pulse_width=10,
    include_arms=False,
    chirp=0,
):
    _ensure_input(start, stop, center, resolution, pulse_width)
    omega0 = center  # unnecessary renaming
    window = (np.sqrt(1 + chirp ** 2) * 8 * np.log(2)) / (pulse_width ** 2)
    lamend = (2 * np.pi * C_LIGHT) / start
    lamstart = (2 * np.pi * C_LIGHT) / stop
    lam = np.arange(lamstart, lamend + resolution, resolution)
    omega = (2 * np.pi * C_LIGHT) / lam
    relom = omega - omega0
    i1 = np.exp(-(relom ** 2) / window)
    i2 = np.exp(-(relom ** 2) / window)
    i = (
        i1
        + i2
        + 2
        * np.cos(
            _disp(relom, GD=GD, GDD=GDD, TOD=TOD, FOD=FOD, QOD=QOD, SOD=SOD) + (omega * delay)
        )
        * np.sqrt(i1 * i2)
    )
    if include_arms:
        return omega, i, i1, i2
    else:
        return omega, i, np.array([]), np.array([])


def generator_wave(
    start,
    stop,
    center,
    delay,
    GD=0,
    GDD=0,
    TOD=0,
    FOD=0,
    QOD=0,
    SOD=0,
    resolution=0.1,
    pulse_width=10,
    include_arms=False,
    chirp=0,
):
    _ensure_input(start, stop, center, resolution, pulse_width)
    omega0 = (2 * np.pi * C_LIGHT) / center
    window = (np.sqrt(1 + chirp ** 2) * 8 * np.log(2)) / (pulse_width ** 2)
    lam = np.arange(start, stop + resolution, resolution)
    omega = (2 * np.pi * C_LIGHT) / lam
    relom = omega - omega0
    i1 = np.exp(-(relom ** 2) / window)
    i2 = np.exp(-(relom ** 2) / window)
    i = (
        i1
        + i2
        + 2
        * np.sqrt(i1 * i2)
        * np.cos(
            _disp(relom, GD=GD, GDD=GDD, TOD=TOD, FOD=FOD, QOD=QOD, SOD=SOD) + (omega * delay)
        )
    )
    if include_arms:
        return lam, i, i1, i2
    else:
        return lam, i, np.array([]), np.array([])
