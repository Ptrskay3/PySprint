"""
Sample generator
"""
import numpy as np

C_LIGHT = 299.793  # nm/fs


def _ensure_input(start, stop, center, resolution, pulseWidth):
    if start >= stop:
        raise ValueError("start value must be less than stop")
    if center < start or center > stop:
        raise ValueError("center must be between start and stop")
    if resolution > (stop - start):
        raise ValueError("resolution is too big")
    if pulseWidth <= 0:
        raise ValueError("Pulse width is strictly positive.")
    else:
        pass


def _disp(x, GD=0, GDD=0, TOD=0, FOD=0, QOD=0):
    return (
        x * GD
        + (GDD / 2) * x ** 2
        + (TOD / 6) * x ** 3
        + (FOD / 24) * x ** 4
        + (QOD / 120) * x ** 5
    )


def generatorFreq(
    start,
    stop,
    center,
    delay,
    GD=0,
    GDD=0,
    TOD=0,
    FOD=0,
    QOD=0,
    resolution=0.1,
    delimiter=",",
    pulseWidth=10,
    includeArms=False,
    chirp=0,
):
    _ensure_input(start, stop, center, resolution, pulseWidth)
    omega0 = center  # unnecessary renaming
    window = (np.sqrt(1 + chirp ** 2) * 8 * np.log(2)) / (pulseWidth ** 2)
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
            _disp(relom, GD=GD, GDD=GDD, TOD=TOD, FOD=FOD, QOD=QOD) + (omega * delay)
        )
        * np.sqrt(i1 * i2)
    )
    if includeArms:
        return omega, i, i1, i2
    else:
        return omega, i, np.array([]), np.array([])


def generatorWave(
    start,
    stop,
    center,
    delay,
    GD=0,
    GDD=0,
    TOD=0,
    FOD=0,
    QOD=0,
    resolution=0.1,
    delimiter=",",
    pulseWidth=10,
    includeArms=False,
    chirp=0,
):
    _ensure_input(start, stop, center, resolution, pulseWidth)
    omega0 = (2 * np.pi * C_LIGHT) / center
    window = (np.sqrt(1 + chirp ** 2) * 8 * np.log(2)) / (pulseWidth ** 2)
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
            _disp(relom, GD=GD, GDD=GDD, TOD=TOD, FOD=FOD, QOD=QOD) + (omega * delay)
        )
    )
    if includeArms:
        return lam, i, i1, i2
    else:
        return lam, i, np.array([]), np.array([])
