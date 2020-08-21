import sys

sys.path.append("..")

import unittest
from unittest.mock import patch

import numpy as np

from pysprint.core._generator import generator_freq, generator_wave, C_LIGHT
from pysprint import Generator as Generator_from_API


class TestGenerator(unittest.TestCase):
    def setUp(self):
        assert C_LIGHT == 299.792458

    def tearDown(self):
        pass

    def test_errors(self):
        with self.assertRaises(ValueError):
            generator_freq(start=1, stop=2, center=3, delay=0)
        with self.assertRaises(ValueError):
            generator_freq(start=5, stop=2, center=3, delay=0)
        with self.assertRaises(ValueError):
            generator_freq(start=1, stop=-1, center=3, delay=0)
        with self.assertRaises(ValueError):
            generator_freq(start=1, stop=3, center=2, delay=0, pulse_width=-20)
        with self.assertRaises(ValueError):
            generator_wave(start=400, stop=800, center=600, delay=0, resolution=1000)

    def test_freq(self):
        a, b, c, d = generator_freq(1, 2, 1.5, delay=0, include_arms=True)
        np.testing.assert_array_equal(c, d)
        assert len(a) == len(b)
        e, f, _, _ = generator_freq(1, 2, 1.5, delay=0)
        assert len(e) == len(f)

    def test_wave(self):
        a, b, c, d = generator_wave(1, 2, 1.5, delay=0, include_arms=True)
        np.testing.assert_array_equal(c, d)
        assert len(a) == len(b)
        e, f, _, _ = generator_wave(1, 2, 1.5, delay=0)
        assert len(e) == len(f)

    @patch("matplotlib.pyplot.show")
    def test_Generator_from_API(self, mock_show):
        g = Generator_from_API(1, 3, 2, 100)
        g.generate_freq()
        g.phase_graph()
        mock_show.assert_called()


if __name__ == "__main__":
    unittest.main()
