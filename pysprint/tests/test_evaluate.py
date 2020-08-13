import unittest
import numpy as np

from pysprint.core import evaluate
from pysprint.core.preprocess import find_peak
from pysprint import Generator, FFTMethod


class TestEvaluate(unittest.TestCase):

    def test_cff(self):
        a = np.arange(100)
        b = np.arange(100)
        with self.assertRaises(KeyError):
            evaluate.cff_method(
                a, b, [], [], ref_point=0, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1]
            )

    def test_ffts_primitive(self):
        # adapted from scipy's unittests
        np.random.seed(1534)
        x = np.random.randn(10) + 1j * np.random.randn(10)
        fr, yf = evaluate.ifft_method(x, x, interpolate=False)
        _, y = evaluate.fft_method(yf, yf)
        np.testing.assert_allclose(y, x)

    def test_ffts_advanced2(self):
        g = Generator(
            2, 2.8, 2.4, delay=1500, GDD=2000, pulse_width=25, resolution=0.01
        )
        g.generate_freq()
        a, b = g.data
        f = FFTMethod(a, b)
        f.ifft()
        f.window(1500, 2920, plot=False)
        f.apply_window()
        f.fft()
        d, _, _ = f.calculate(order=2, reference_point=2.4)
        np.testing.assert_array_almost_equal(d, [1500.01, 1999.79], decimal=2)

    def test_ffts_advanced1(self):
        g = Generator(2, 2.8, 2.4, delay=1500, GD=200, pulse_width=25, resolution=0.01)
        g.generate_freq()
        a, b = g.data
        f = FFTMethod(a, b)
        f.ifft()
        f.window(1700, 3300, plot=False)
        f.apply_window()
        f.fft()
        d, _, _ = f.calculate(order=1, reference_point=2.4)
        np.testing.assert_array_almost_equal(d, [1699.99], decimal=2)

    def test_ffts_advanced3(self):
        g = Generator(
            2, 2.8, 2.4, delay=1500, TOD=40000, pulse_width=25, resolution=0.01
        )
        g.generate_freq()
        a, b = g.data
        f = FFTMethod(a, b)
        f.ifft()
        f.window(2500, 4830, window_order=12, plot=False)
        f.apply_window()
        f.fft()
        d, _, _ = f.calculate(order=3, reference_point=2.4)
        np.testing.assert_array_almost_equal(d, [1500.03, 0.03, 39996.60], decimal=2)

    def test_ffts_advanced4(self):
        g = Generator(
            2,
            2.8,
            2.4,
            delay=1500,
            GDD=2000,
            FOD=-100000,
            pulse_width=25,
            resolution=0.01,
        )
        g.generate_freq()
        a, b = g.data
        f = FFTMethod(a, b)
        f.ifft()
        f.window(1500, 1490, window_order=8, plot=False)
        f.apply_window()
        f.fft()
        d, _, _ = f.calculate(order=4, reference_point=2.4)
        np.testing.assert_array_almost_equal(
            d, [1500.00, 1999.95, -0.21, -99995.00], decimal=1
        )

    def test_ffts_advanced5(self):
        g = Generator(
            2, 2.8, 2.4, delay=1500, QOD=900000, pulse_width=25, resolution=0.01,
        )
        g.generate_freq()
        a, b = g.data
        f = FFTMethod(a, b)
        f.ifft()
        f.window(1600, 2950, window_order=12, plot=False)
        f.apply_window()
        f.fft()
        d, _, _ = f.calculate(order=5, reference_point=2.4)
        np.testing.assert_array_almost_equal(
            d, [1499.96, -0.14, 7.88, 15.99, 898920.79], decimal=1
        )

    def test_windowing(self):
        a, b = np.loadtxt("test_window.txt", unpack=True, delimiter=",")
        y_data = evaluate.cut_gaussian(a, b, 2.5, 0.2, 6)
        assert len(b) == len(y_data)
        np.testing.assert_almost_equal(y_data[0], 0)
        np.testing.assert_almost_equal(y_data[-1], 0)
        np.testing.assert_almost_equal(np.median(y_data), np.median(b), decimal=2)

    def test_spp(self):
        pass


if __name__ == "__main__":
    unittest.main()
