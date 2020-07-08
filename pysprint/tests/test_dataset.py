import sys
import collections
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

from pysprint import Dataset
from pysprint.utils.exceptions import DatasetError


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(1, 1000, 1)
        self.y = np.sin(self.x)

    def tearDown(self):
        pass

    def test_constructor(self):
        ifg = Dataset(self.x, self.y)
        np.testing.assert_array_equal(ifg.data["x"], self.x)
        np.testing.assert_array_equal(ifg.data["y"], self.y)

    def test_prediction(self):
        ifg = Dataset([150, 200], [3, 4])
        assert ifg.probably_wavelength == True
        ifg = Dataset([1, 2], [3, 4])
        assert ifg.probably_wavelength == False

    def test_dtypes(self):
        ifg = Dataset([15, 4], [3, 4])
        assert type(ifg.x) == np.ndarray
        assert type(ifg.y) == np.ndarray
        with self.assertRaises(DatasetError):
            ifg = Dataset(["das", 6541], [1, 2])
        with self.assertRaises(DatasetError):
            ifg = Dataset([2, 6541], ["das", 2])

    def test_safe_casting(self):
        ifg = Dataset([15, 4], [3, 4], [14, 54], [45, 51])
        x, y, ref, sam = ifg._safe_cast()
        assert x is not ifg.x
        np.testing.assert_array_equal(x, ifg.x)
        assert y is not ifg.y
        np.testing.assert_array_equal(y, ifg.y)
        assert ref is not ifg.ref
        np.testing.assert_array_equal(ref, ifg.ref)
        assert sam is not ifg.sam
        np.testing.assert_array_equal(sam, ifg.sam)

    def test_rawparsing(self):
        ifg = Dataset.parse_raw('test_rawparsing.trt')
        assert issubclass(ifg.meta.__class__, collections.abc.Mapping)
        with self.assertRaises(OSError):
            ifg = Dataset.parse_raw(546)

    def test_data(self):
        ifg = Dataset(self.x, self.y)
        assert isinstance(ifg.data, pd.DataFrame)

    @patch("matplotlib.pyplot.show")
    def test_plotting(self, mock_show):
        ifg = Dataset(self.x, self.y)
        ifg.show()
        mock_show.assert_called()

    def test_chdomain(self):
        ifg = Dataset(self.x, self.y)
        before = ifg.x
        ifg.chdomain()
        ifg.chdomain()
        after = ifg.x
        np.testing.assert_array_almost_equal(before, after)

    @pytest.mark.skip(reason="Fails on azure, should be fixed ")
    @patch('matplotlib.pyplot.show')
    def test_normalize(self, mock_show):
    	ifg = Dataset(self.x, self.y)
    	ifg.normalize()
    	mock_show.assert_called()

    @patch("matplotlib.pyplot.show")
    def test_sppeditor(self, mock_show):
        ifg = Dataset(self.x, self.y)
        ifg.open_SPP_panel()
        mock_show.assert_called()

    def test_spp_setting(self):
        ifg = Dataset(self.x, self.y)
        delay = 100
        positions = [200, 300]
        ifg.set_SPP_data(delay, positions)
        delay, position = ifg.emit()
        assert len(delay) == len(position)
        np.testing.assert_array_equal(delay, np.array([100, 100]))
        np.testing.assert_array_equal(position, positions)

    def test_slicing_inplace(self):
        ifg = Dataset(self.x, self.y)
        ifg.slice(400, 700)

        assert np.min(ifg.x) > 399
        assert np.max(ifg.x) < 701


    def test_slicing_non_inplace(self):
        ifg = Dataset(self.x, self.y)
        new_ifg = ifg.slice(400, 700, inplace=False)

        assert np.min(new_ifg.x) > 399
        assert np.max(new_ifg.x) < 701

        assert not np.min(ifg.x) > 399
        assert not np.max(ifg.x) < 701

if __name__ == "__main__":
    unittest.main()
