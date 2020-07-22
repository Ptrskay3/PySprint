import unittest
from unittest.mock import patch

import numpy as np

import pysprint
from pysprint import FFTMethod, Generator
from pysprint.core.fft_tools import (
    find_roi,
    find_center,
    _ensure_window_at_origin,
    predict_fwhm,
)


class TestFFTAuto(unittest.TestCase):
    def test_find_roi(self):
        x, y = find_roi(np.linspace(-10, 10, 10000), np.linspace(-10, 10, 10000))
        np.testing.assert_almost_equal(min(x), 0, decimal=2)
        np.testing.assert_almost_equal(min(y), 0, decimal=2)

    def test_find_center1(self):
        g = Generator(1, 4, 3, 1500)
        g.generate_freq()

        a = FFTMethod(*g.data)
        a.ifft()

        x, y = find_roi(a.x, a.y)

        x_peak, _ = find_center(x, y, n_largest=4)

        assert 1495 < x_peak < 1505

    def test_find_center2(self):
        g = Generator(1, 4, 3, 100)
        g.generate_freq()

        a = FFTMethod(*g.data)
        a.ifft()

        x, y = find_roi(a.x, a.y)

        x_peak, _ = find_center(x, y, n_largest=4)

        assert 95 < x_peak < 105

    def test_find_center3(self):
        g = Generator(1, 4, 3, 7000)
        g.generate_freq()

        a = FFTMethod(*g.data)
        a.ifft()

        x, y = find_roi(a.x, a.y)

        x_peak, _ = find_center(x, y, n_largest=4)

        assert 6950 < x_peak < 7050

    def test_window1(self):
        is_fine, val = _ensure_window_at_origin(0, 200, 4, 10, tol=1e-3)
        assert not is_fine

    def test_window2(self):
        is_fine, val = _ensure_window_at_origin(500, 200, 8, 10, tol=1e-3)
        assert is_fine

    def test_window3(self):
        # test against ambigous overlaps
        is_fine, val = _ensure_window_at_origin(500, 249, 4, 10, tol=1e-3)
        assert is_fine

    def test_window4(self):
        # test against ambigous overlaps
        is_fine, val = _ensure_window_at_origin(500, 251, 4, 10, tol=1e-3)
        assert is_fine

    #  Obviously we need more verbose tests for that later.

    def test_fwhm_detection(self):
        g = Generator(1, 4, 3, 1000)
        g.generate_freq()

        a = FFTMethod(*g.data)
        a.ifft()

        x, y = find_roi(a.x, a.y)

        cent, win_size, order = predict_fwhm(
            x, y, 1000, 10, prefer_high_order=True, tol=1e-3
        )
        assert cent > 1000
        assert win_size > 2000

    @patch("matplotlib.pyplot.show")
    def test_autorun(self, mck):
        g = Generator(1, 4, 3, 1000)
        g.generate()

        f = FFTMethod(*g.data)
        phase = f.autorun(enable_printing=True)
        assert isinstance(phase, pysprint.core.phase.Phase)
        mck.assert_called()


if __name__ == "__main__":
    unittest.main()
