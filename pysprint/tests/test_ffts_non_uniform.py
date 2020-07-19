import unittest
from unittest import TestCase
from pysprint.core.nufft import nuifft

import numpy as np


class TestNuifft(TestCase):
    def setUp(self):
        rng = np.random.RandomState(151560)
        self.x = 100 * rng.rand(100)
        self.y = np.exp(1j * self.x)
        self.gl = len(self.x)
        self.eps = 1e-12

    def test_nuifft(self):
        dft = nuifft(self.x, self.y, self.gl, exponent="positive")
        fft = nuifft(self.x, self.y, self.gl, exponent="positive", epsilon=self.eps)
        np.testing.assert_allclose(dft, fft, rtol=self.eps ** 0.95)

        dft = nuifft(self.x, self.y.real, self.gl, exponent="positive")
        fft = nuifft(
            self.x, self.y.real, self.gl, exponent="positive", epsilon=self.eps
        )
        np.testing.assert_allclose(dft, fft, rtol=self.eps ** 0.95)


if __name__ == "__main__":
    unittest.main()
