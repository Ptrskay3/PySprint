import os
import collections
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

from pysprint import Dataset
from pysprint.utils.exceptions import DatasetError, PySprintWarning, InterpolationWarning


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
        assert ifg.probably_wavelength
        ifg = Dataset([1, 2], [3, 4])
        assert not ifg.probably_wavelength

    def test_dtypes(self):
        ifg = Dataset([15, 4], [3, 4])
        assert type(ifg.x) == np.ndarray
        assert type(ifg.y) == np.ndarray
        with self.assertRaises(DatasetError):
            Dataset(["das", 6541], [1, 2])
        with self.assertRaises(DatasetError):
            Dataset([2, 6541], ["das", 2])

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
        ifg = Dataset.parse_raw(
            "test_rawparsing.trt", skiprows=8, meta_len=5, decimal=",", sep=";"
        )
        assert issubclass(ifg.meta.__class__, collections.abc.Mapping)
        with self.assertRaises(OSError):
            Dataset.parse_raw(546)

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

    @pytest.mark.skipif("TF_BUILD" in os.environ, reason="Fails on Azure")
    @patch("matplotlib.pyplot.show")
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

    def test_slicing_noinplace(self):
        ifg = Dataset(self.x, self.y)
        new_ifg = ifg.slice(400, 700, inplace=False)

        assert np.min(new_ifg.x) > 399
        assert np.max(new_ifg.x) < 701

        assert not np.min(ifg.x) > 399
        assert not np.max(ifg.x) < 701

    def test_chdomain_noinplace(self):
        ifg = Dataset(self.x, self.y)
        before = ifg.x
        f = ifg.chdomain(inplace=False)
        g = f.chdomain(inplace=False)
        after = g.x
        np.testing.assert_array_almost_equal(before, after)


@pytest.mark.parametrize("pos", [(1, 2), {4, 5}, [4, 5], np.array([1, 4])])
def test_SPP_position_setter_valid_options(pos):
    ifg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    ifg.positions = pos


@pytest.mark.parametrize("pos", ["string", [1500, 4500], [100, "string"]])
def test_SPP_position_setter_invalid_options(pos):
    ifg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    with pytest.raises(ValueError):
        ifg.positions = pos


def test_SPP_position_setter_invalid_single():
    ifg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    with pytest.raises(ValueError):
        ifg.positions = -2000


@patch("matplotlib.pyplot.show")
def test_phase_plot(m):
    fg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    fg._dispersion_array = np.array([1, 2, 3, 4, 5])
    fg.phase_plot()
    m.assert_called()


@pytest.mark.parametrize("to", ["ghz", "thz"])
def test_chrange_basic(to):
    fg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    fg.chrange(current_unit="PHz", target_unit=to)
    if to == "thz":
        np.testing.assert_array_almost_equal(np.min(fg.x), 1000)
    elif to == "ghz":
        np.testing.assert_array_almost_equal(np.min(fg.x), 1000000)


@pytest.mark.parametrize("to", ["ghz", "thz"])
def test_chrange_basic_noninplace(to):
    fg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    ifg = fg.chrange(current_unit="PHz", target_unit=to, inplace=False)
    if to == "thz":
        np.testing.assert_array_almost_equal(np.min(ifg.x), 1000)
    elif to == "ghz":
        np.testing.assert_array_almost_equal(np.min(ifg.x), 1000000)
    np.testing.assert_array_equal(fg.x, np.arange(1, 1000, 1))


def test_chrange_errors():
    fg = Dataset(np.arange(1, 1000, 1), np.sin(np.arange(1, 1000, 1)))
    with pytest.raises(ValueError):
        fg.chrange(current_unit="PHz", target_unit="undefined")


def test_blank_repr():
    expected = [
        "Dataset",
        '----------',
        'Parameters',
        '----------',
        'Datapoints: 199',
        'Predicted domain: frequency',
        'Range: from 1.000 to 199.000 PHz',
        'Normalized: True',
        'Delay value: Not given',
        'SPP position(s): Not given',
        '----------------------------',
        'Metadata extracted from file',
        '----------------------------',
        '{}'
    ]
    x_ = np.arange(200)
    y_ = np.linspace(0, 1, 199)

    d = Dataset(x_, y_, errors='force')

    string = d.__repr__().split("\n")
    assert string == expected


def test_blank_repr_2():
    expected = [
        "Dataset",
        '----------',
        'Parameters',
        '----------',
        'Datapoints: 199',
        'Predicted domain: frequency',
        'Range: from 1.000 to 199.000 PHz',
        'Normalized: True',
        'Delay value: 100 fs',
        'SPP position(s): Not given',
        '----------------------------',
        'Metadata extracted from file',
        '----------------------------',
        '{}'
    ]
    x_ = np.arange(200)
    y_ = np.linspace(0, 1, 199)

    d = Dataset(x_, y_, errors='force')
    d.delay = 100

    string = d.__repr__().split("\n")
    assert string == expected


def test_blank_repr_3():
    expected = [
        "Dataset",
        '----------',
        'Parameters',
        '----------',
        'Datapoints: 199',
        'Predicted domain: frequency',
        'Range: from 0.000 to 1.000 PHz',
        'Normalized: True',
        'Delay value: 100 fs',
        'SPP position(s): 0.5 PHz',
        '----------------------------',
        'Metadata extracted from file',
        '----------------------------',
        '{}'
    ]
    x_ = np.linspace(0, 1, 199)

    d = Dataset(x_, x_)
    d.delay = 100
    d.positions = 0.5
    string = d.__repr__().split("\n")
    assert string == expected


def test_raw_repr():
    string = [
        'Dataset',
        '----------',
        'Parameters',
        '----------',
        'Datapoints: 2633',
        'Predicted domain: wavelength',
        'Range: from 360.500 to 1200.250 nm',
        'Normalized: False',
        'Delay value: Not given',
        'SPP position(s): Not given',
        '----------------------------',
        'Metadata extracted from file',
        '----------------------------',
        "{",
        '    "Average": "1 scans",',
        '    "Data measured with spectrometer name": "1107006U1",',
        '    "Integration time": "2,00 ms",',
        '    "Nr of pixels used for smoothing": "0",',
        '    "comment": "m_ifg 8,740",',
        '    "unparsed": [',
        '        [',
        '            "Timestamp [10 microsec ticks]256373440"',
        '        ]',
        '    ]',
        '}'
    ]

    ifg = Dataset.parse_raw("test_rawparsing.trt", skiprows=8, meta_len=5, decimal=",", sep=";")
    assert ifg.__repr__().split("\n") == string


def test_arm_warning():
    with pytest.warns(PySprintWarning):
        Dataset([2], [3], [4])


def test_scale_up():
    x_ = np.linspace(0, 1, 199)

    d = Dataset(x_, x_)
    d.scale_up()

    np.testing.assert_array_almost_equal(d.y, 2 * (x_ - 0.5))
    np.testing.assert_array_almost_equal(d.y_norm, 2 * (x_ - 0.5))


def test_GD_lookup():
    x_ = np.linspace(0, 1, 199)
    d = Dataset(x_, x_)
    with pytest.warns(UserWarning):
        d.GD_lookup(2, engine="fft")


@pytest.mark.parametrize("val", [-1, 1, 200])
def test_wave2freq(val):
    before = val
    after = Dataset.wave2freq(Dataset.freq2wave(val))
    assert before == after


def test_freq2wave():
    with pytest.raises(ZeroDivisionError):
        Dataset.wave2freq(Dataset.freq2wave(0))


def test_is_normalized():
    x_ = np.linspace(0, 1, 199)
    d = Dataset(x_, x_)
    assert d.is_normalized


def test_is_not_normalized():
    x_ = np.linspace(0, 10, 199)
    d = Dataset(x_, x_)
    assert not d.is_normalized


@pytest.mark.parametrize("func", ["savgol_fil", "convolution"])
def test_specfuncs(func):
    x_ = np.linspace(0, 1, 199)
    d = Dataset(x_, x_)
    with pytest.warns(InterpolationWarning):
        getattr(d, func)(5, 3)
    np.testing.assert_array_equal(d.y, d.y_norm)


if __name__ == "__main__":
    unittest.main()
