import unittest

import pytest
import numpy as np

from pysprint.utils import (
    _handle_input,
    print_disp,
    fourier_interpolate,
    pad_with_trailing_zeros,
    find_nearest,
    measurement,
)


class TestMisc(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(100)
        self.y = np.sin(self.x)
        self.ref = np.ones_like(self.x)
        self.sam = np.ones_like(self.x)

        self._y = (self.y - self.sam - self.ref) / (2 * np.sqrt(self.sam * self.ref))

    def test_input_handling1(self):
        x, y = _handle_input(self.x, self.y, [], [])
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self.y)

    def test_input_handling2(self):
        x, y = _handle_input(self.x, self.y, self.ref, self.sam)
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self._y)

    def test_input_handling3(self):
        x, y = _handle_input(self.x, self.y, self.ref, [])
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self.y)

    def test_input_handling4(self):
        with self.assertRaises(ValueError):
            _handle_input(self.x, [], [], [])

    def test_input_handling5(self):
        with self.assertRaises(ValueError):
            _handle_input([], self.y, [], [])

    @pytest.mark.skip(reason="IPython.display treats this another way")
    def test_pprint(self):
        """source : https://stackoverflow.com/a/4220278/11751294"""
        import sys
        from io import StringIO

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out

            @print_disp
            def calculation():
                return [0, 1], [0, 1], ""

            calculation()
            output = out.getvalue().strip()
            assert (
                output
                == """GD = 0.00000 ± 0.00000 fs^1
GDD = 1.00000 ± 1.00000 fs^2"""
            )
        finally:
            sys.stdout = saved_stdout

    def test_fourier_interpol(self):
        x, _ = fourier_interpolate(np.geomspace(10, 1000), np.geomspace(10, 1000))
        np.testing.assert_array_equal(x, np.linspace(10, 1000))

    def test_padding(self):
        arr = np.array([1, 5, 6])
        res = pad_with_trailing_zeros(arr, shape=8)
        assert len(res) == 8

    def test_find_nearest(self):
        x = np.array([1, 2, 4, 9])
        el, idx = find_nearest(x, 10)
        assert el == 9
        assert idx == 3

    def test_measurement(self):
        a = np.array([123.783, 121.846, 122.248, 125.139, 122.569])
        mean, interval = measurement(a, 0.99, silent=True)
        assert mean == 123.117
        assert interval == (120.35397798230359, 125.88002201769642)


if __name__ == "__main__":
    unittest.main()
