import numpy as np

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
    mck.assert_called()
