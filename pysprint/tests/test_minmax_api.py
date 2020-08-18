import os
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import matplotlib.pyplot as plt

from pysprint import MinMaxMethod


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(1, 1000, 1)
        self.y = np.sin(self.x)

    def tearDown(self):
        pass

    @pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure Pipelines fails this.")
    @patch("matplotlib.pyplot.show")
    def test_edit_session(self, mock_show):
        ifg = MinMaxMethod(self.x, self.y)
        ifg.init_edit_session(engine="cwt")
        ifg.init_edit_session(engine="slope")
        ifg.init_edit_session(engine="normal")
        plt.close("all")
        mock_show.assert_called()

    def test_edit_session2(self):
        ifg = MinMaxMethod(self.x, self.y)
        with self.assertRaises(ValueError):
            ifg.init_edit_session(engine="dfssdfasdf")

    def test_edit_session3(self):
        ifg = MinMaxMethod(self.x, self.y)
        with self.assertRaises(TypeError):
            ifg.init_edit_session(engine="normal", invalidkwarg=3)

    @pytest.mark.skipif("TF_BUILD" in os.environ, reason="Fails on azure.")
    @patch("matplotlib.pyplot.show")
    def test_edit_session4(self, mock_show):
        ifg = MinMaxMethod(self.x, self.y)
        ifg.init_edit_session()
        plt.close("all")
        mock_show.assert_called()


def test_phase_build():
    m = MinMaxMethod(np.arange(10), np.arange(10))
    m.xmin = np.arange(0, 10)[::2]
    m.xmax = np.arange(0, 10)[1::2]

    m.build_phase(reference_point=0)

    np.testing.assert_array_equal(np.arange(10), m.phase.x)
    np.testing.assert_array_almost_equal(
        np.array(
            [
                -3.141593,
                -3.141593,
                -6.283185,
                -9.424778,
                -12.566371,
                -15.707963,
                -18.849556,
                -21.991149,
                -25.132741,
                -28.274334
            ]
        ),
        m.phase.y
    )


def test_calculate():
    m = MinMaxMethod(np.arange(10), np.arange(10))
    m.xmin = np.arange(0, 10)[::2]
    m.xmax = np.arange(0, 10)[1::2]

    d, ds, _ = m.calculate(reference_point=0, order=1)

    np.testing.assert_array_almost_equal(d, [-2.970233])

    np.testing.assert_array_equal(np.arange(10), m.phase.x)
    np.testing.assert_array_almost_equal(
        np.array(
            [
                -3.141593,
                -3.141593,
                -6.283185,
                -9.424778,
                -12.566371,
                -15.707963,
                -18.849556,
                -21.991149,
                -25.132741,
                -28.274334
            ]
        ),
        m.phase.y
    )


if __name__ == "__main__":
    unittest.main()
