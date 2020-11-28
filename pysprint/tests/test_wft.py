from unittest.mock import patch
import importlib

import pytest
import numpy as np

from pysprint import Generator, WFTMethod
from pysprint.utils import NotCalculatedException


@pytest.mark.slow
@patch('matplotlib.pyplot.show')
def test_basic(mck):
    g = Generator(
        1,
        3,
        2,
        3000,
        GDD=400,
        TOD=4000,
        FOD=4000,
        QOD=50000,
        pulse_width=5,
        resolution=0.01
    )

    g.generate()

    f = WFTMethod(*g.data)
    f.add_window_linspace(1.25, 2.75, 350, fwhm=0.017)

    d, _, _ = f.calculate(
        reference_point=2, order=5, fastmath=False, silent=True, parallel=False
    )

    np.testing.assert_array_almost_equal(
        d, [2999.15, 399.94, 3998.03, 3991.45, 49894.96], decimal=1
    )

    f.heatmap()
    f.show()
    mck.assert_called()


def test_window_basic():
    w = WFTMethod(np.arange(100), np.arange(100))
    w.add_window(20, fwhm=1)
    with pytest.raises(TypeError):
        w.add_window(515)
    w.remove_window_at(20)
    w.add_window_linspace(1, 20, 19, fwhm=1)
    w.remove_window_interval(1, 20)
    assert len(w.windows) == 0


@patch('matplotlib.pyplot.show')
def test_window_generic(mock_show):
    w = WFTMethod(np.arange(100), np.arange(100))
    w.add_window_generic(np.random.normal(50, 10, size=50), fwhm=50)
    assert len(w.windows) == 50
    w.view_windows()
    w.reset_state()
    assert len(w.windows) == 0
    with pytest.raises(NotCalculatedException):
        w.heatmap()
    with pytest.raises(NotCalculatedException):
        w.get_GD()


# @pytest.mark.skipif(
#     importlib.util.find_spec('dask') is None,
#     reason="dask is required"
# )
# @pytest.mark.slow
# def test_basic_parallel():
#     g = Generator(
#         1,
#         3,
#         2,
#         3000,
#         GDD=400,
#         TOD=4000,
#         FOD=4000,
#         QOD=50000,
#         pulse_width=5,
#         resolution=0.01
#     )

#     g.generate()

#     f = WFTMethod(*g.data)
#     f.add_window_linspace(1.25, 2.75, 350, fwhm=0.017)

#     d, _, _ = f.calculate(
#         reference_point=2, order=5, fastmath=False, parallel=True, silent=True
#     )

#     np.testing.assert_array_almost_equal(
#         d, [2999.16249, 399.94162, 3998.03069, 3991.45663, 49894.96710], decimal=1
#     )
