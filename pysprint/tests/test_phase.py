import pytest
import numpy as np
import matplotlib.pyplot as plt

from unittest.mock import patch
from pysprint.core.phase import Phase


def test_phase_gd_mode():
    x = np.arange(100)
    y = np.arange(100)

    phase = Phase(x, y, GD_mode=True)
    d, ds, st = phase.fit(reference_point=50, order=2)
    np.testing.assert_array_almost_equal(d, [50, 1])
    assert phase.order == 1
    assert phase.dispersion_order == 2


@patch("matplotlib.pyplot.show")
def test_plots(mck):
    x = np.arange(100)
    y = np.arange(100)

    phase = Phase(x, y)
    phase.plot()
    plt.show()
    mck.assert_called()


def test_errorplot():
    x = np.arange(100)
    y = np.arange(100)

    phase = Phase(x, y)
    phase.fit(2, 2)
    phase.errorplot()
    phase.errorplot(percent=True)


def test_errorplot2():
    x = np.arange(100)
    y = np.arange(100)

    phase = Phase(x, y)
    with pytest.raises(ValueError):
        phase.errors
    with pytest.raises(ValueError):
        phase.errorplot()


def constructor1():
    Phase.from_coeff([1, 2, 3, 4])
    Phase.from_coeff([1, 2, 3, 4], domain=np.arange(5611))


def constructor2():
    Phase.from_disperion_array([1, 2, 3, 4])
    Phase.from_disperion_array([1, 2, 3, 4], domain=np.arange(5611))
