import pytest

import numpy as np

from pysprint.core.methods import SPPMethod
from pysprint.core.bases import Dataset


@pytest.fixture()
def construct_ifg_sequence():
    x = np.arange(1000)
    y = np.sin(x)

    # setting multiple SPP
    d1 = Dataset(x, y)
    d1.set_SPP_data(10, [40, 45])

    # simple
    d2 = Dataset(x, y)
    d2.set_SPP_data(12, 13)

    # setting 3 SPP
    d3 = Dataset(x, y)
    d3.set_SPP_data(14, [14, 500, 43])

    # setting SPP directly
    d4 = Dataset(x, y)
    d4.delay = 40
    d4.positions = np.array([45.8, 54])

    # handle duplicates
    d5 = Dataset(x, y)
    d5.delay = 40
    d5.positions = [5.8, 40]

    # ambiguous positions
    d6 = Dataset(x, y)
    d6.delay = 40
    d6.positions = 45.8, 54

    return d1, d2, d3, d4, d5, d6


def test_collection(construct_ifg_sequence):
    """
    Here we test the basic functionality of calculate_from_ifg.
    """
    d1, d2, d3, d4, _, _ = construct_ifg_sequence
    disp, _, _ = SPPMethod.calculate_from_ifg(
        [d1, d2, d3, d4], reference_point=2, order=5
    )
    np.testing.assert_almost_equal(-108.59671, disp[0], decimal=5)
    np.testing.assert_almost_equal(12.27231, disp[1], decimal=5)
    np.testing.assert_almost_equal(-0.76698, disp[2], decimal=5)
    np.testing.assert_almost_equal(0.02171, disp[3], decimal=5)
    np.testing.assert_almost_equal(-0.00014, disp[4], decimal=5)


def test_spp_typeerror():
    with pytest.raises(TypeError):
        SPPMethod.calculate_from_ifg([10, 11], reference_point=2, order=2)


def test_duplicate_entries(construct_ifg_sequence):
    """
    Here we test that duplicated delay values (apart from multiple SPP positions)
    are correctly identified and ValueError is raised.
    """
    d1, d2, d3, d4, d5, d6 = construct_ifg_sequence
    with pytest.raises(ValueError):
        SPPMethod.calculate_from_ifg([d1, d2, d3, d4, d5], reference_point=2, order=4)


def test_ambiguous_positions(construct_ifg_sequence):
    """
    Here we test if ambiguous positions (defining them as tuple) are handled correctly.
    """
    d1, d2, d3, d4, d5, d6 = construct_ifg_sequence
    disp, _, _ = SPPMethod.calculate_from_ifg(
        [d1, d2, d3, d6], reference_point=2, order=5
    )
    np.testing.assert_almost_equal(-108.59671, disp[0], decimal=5)
    np.testing.assert_almost_equal(12.27231, disp[1], decimal=5)
    np.testing.assert_almost_equal(-0.76698, disp[2], decimal=5)
    np.testing.assert_almost_equal(0.02171, disp[3], decimal=5)
    np.testing.assert_almost_equal(-0.00014, disp[4], decimal=5)
