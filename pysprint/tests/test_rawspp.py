import unittest
from unittest.mock import patch

import numpy as np

from pysprint.core.evaluate import spp_method
from pysprint import SPPMethod


class TestEvaluate(unittest.TestCase):
    def setUp(self):

        self.delays = np.array(
            [
                -1700.0,
                -1700.0,
                -1500.0,
                -1500.0,
                -1300.0,
                -1300.0,
                -1100.0,
                -1100.0,
                -900.0,
                -900.0,
                -700.0,
                -700.0,
                -500.0,
                -500.0,
                -300.0,
                -300.0,
                -100.0,
                -100.0,
                100.0,
                100.0,
                300.0,
            ]
        )

        self.delays_invalid = np.array(
            [
                -1700.0,
                -1700.0,
                -1500.0,
                -1500.0,
                -1300.0,
                -1300.0,
                -1100.0,
                -1100.0,
                -900.0,
                -900.0,
                -700.0,
                -700.0,
                -500.0,
                np.nan,
                -500.0,
                -300.0,
                -300.0,
                -100.0,
                -100.0,
                100.0,
                100.0,
                300.0,
                900,
            ]
        )

        self.omegas = np.array(
            [
                2.15686148,
                2.55201208,
                2.16708089,
                2.54264428,
                2.17900354,
                2.53072164,
                2.19092619,
                2.51965061,
                2.20284884,
                2.50772796,
                2.21732634,
                2.49325046,
                2.23180384,
                2.47536649,
                2.24968781,
                2.45918575,
                2.27097825,
                2.43874693,
                2.29993325,
                2.40894031,
                2.33825604,
            ]
        )

    def test_spp_in_core(self):
        x, y, d, ds, _ = spp_method(
            self.delays, self.omegas, fit_order=2, ref_point=2.355
        )
        np.testing.assert_array_equal(
            d, [-258.84297727172856, 21.572879102888976, 100426.4054547129, 0, 0],
        )

    def test_spp_exceptions(self):
        with self.assertRaises(ValueError):
            x, y, d, ds, _ = spp_method(
                self.delays, self.omegas, ref_point=0, fit_order=8
            )
        with self.assertRaises(ValueError):
            x, y, d, ds, _ = spp_method(
                self.delays, self.omegas, ref_point=0, fit_order=3.5
            )

    def test_spp_invalid(self):
        with self.assertRaises(ValueError):
            x, y, d, ds, _ = spp_method(
                self.delays_invalid, self.omegas, fit_order=2, ref_point=2.355
            )

    def test_spp_not_enough_vars(self):
        with self.assertRaises(TypeError):
            x, y, d, ds, _ = spp_method(
                np.array([1, 2, 3]), np.array([4, 5, 6]), fit_order=4, ref_point=2.355,
            )

    def test_spp_from_raw_api(self):
        with patch("matplotlib.pyplot.show") as mock_show:
            ifgs = SPPMethod("")
            ifgs.listen(self.delays, self.omegas)
            d, ds, _ = ifgs.calculate(2.355, 3, show_graph=True)
            np.testing.assert_array_equal(
                d, [-258.84297727172856, 21.572879102888976, 100426.4054547129],
            )
            mock_show.assert_called()


if __name__ == "__main__":
    unittest.main()
