import unittest
import numpy as np

from pysprint.core.preprocess import *


class TestEdit(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_savgol(self):
        x, y, v, w = np.loadtxt("test_arms.txt", delimiter=",", unpack=True)
        a, b = savgol(x, y, v, w, window=10, order=3)
        c, d = savgol(x, y, v, w, window=11, order=3)
        assert len(a) == len(b)
        assert len(d) == len(d)
        with self.assertRaises(ValueError):
            savgol(x, y, v, w, window=1, order=3)

    def test_peak(self):
        x, y = np.loadtxt("test_peak.txt", delimiter=",", unpack=True)
        a, b, c, d = find_peak(x, y, [], [], threshold=0.01, pro_min=0.5, pro_max=0.5)
        assert len(a) == len(b)
        assert len(c) == len(d)
        for val in b:
            assert abs(val) > 0.01
        for val in d:
            assert abs(val) > 0.01

    def test_interpolate(self):
        x, y, v, w = np.loadtxt("test_arms.txt", delimiter=",", unpack=True)
        a, b = interpolate_data(x, y, v, w)
        assert len(a) == len(b)
        np.random.seed(1000)
        idx1 = np.random.randint(0, len(a))
        idx2 = np.random.randint(0, len(a))
        i = abs(a[idx1] - a[idx1 - 1])
        j = abs(a[idx2] - a[idx2 - 1])
        np.testing.assert_almost_equal(i, j)
        with self.assertRaises(TypeError):
            interpolate_data({"a": "1", "b": "2"}, x, y, v, window=1, order=3)

    def test_cut(self):
        x, y, v, w = np.loadtxt("test_arms.txt", delimiter=",", unpack=True)
        a, b = cut_data(x, y, v, w, start=2.2, stop=2.8)
        assert len(a) == len(b)
        np.testing.assert_almost_equal(min(a), 2.2, decimal=2)
        np.testing.assert_almost_equal(max(a), 2.8, decimal=2)
        with self.assertRaises(ValueError):
            cut_data(x, [], v, w)

    def test_convolution(self):
        x, y, v, w = np.loadtxt("test_arms.txt", delimiter=",", unpack=True)
        a, b = convolution(x, y, v, w, len(x), standev=200)
        assert len(a) == len(b)
        with self.assertRaises(ValueError):
            convolution(x, [], v, w, 10, standev=-541)


if __name__ == "__main__":
    unittest.main()
