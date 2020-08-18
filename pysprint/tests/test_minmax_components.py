import numpy as np
import pytest

from pysprint.core.evaluate import (
    is_inside, _split_on_SPP, _build_single_phase_data, min_max_method)


def test_is_inside():
    x = np.linspace(10, 12, 10)
    assert not is_inside(20, x)
    assert is_inside(11, x)
    assert not is_inside(12, x)
    assert not is_inside(10, x)


def test_splitting_single():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, 2)
    np.testing.assert_array_equal(splitted[0], np.array([-10.]))
    np.testing.assert_array_equal(splitted[-1], [3, 4, 5, 6, 9.1])
    assert len(splitted) == 2


def test_splitting_multiple():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, [2, 4])
    np.testing.assert_array_equal(splitted[0], np.array([-10.]))
    np.testing.assert_array_equal(splitted[1], np.array([3]))
    np.testing.assert_array_equal(splitted[-1], [5, 6, 9.1])
    assert len(splitted) == 3


def test_splitting_single_outside():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, [-20])
    np.testing.assert_array_equal(splitted, [x])
    assert len(splitted) == 1


def test_splitting_multiple_outside():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, [-20, 30])
    np.testing.assert_array_equal(splitted, [x])
    assert len(splitted) == 1


def test_splitting_single_nearest():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, 2.2)
    np.testing.assert_array_equal(splitted[0], np.array([-10.]))
    np.testing.assert_array_equal(splitted[-1], [3, 4, 5, 6, 9.1])
    assert len(splitted) == 2


def test_splitting_multiple_nearest():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, [2.1, 3.9])
    np.testing.assert_array_equal(splitted[0], np.array([-10.]))
    np.testing.assert_array_equal(splitted[1], np.array([3]))
    np.testing.assert_array_equal(splitted[-1], [5, 6, 9.1])
    assert len(splitted) == 3


def test_splitting_multiple_nearest_outside():
    x = np.array([-10, 2, 3, 4, 5, 6, 9.1])
    splitted = _split_on_SPP(x, [-20, 3.9])
    np.testing.assert_array_equal(splitted[0], np.array([-10, 2, 3]))
    np.testing.assert_array_equal(splitted[-1], [5, 6, 9.1])
    assert len(splitted) == 2


def test_build_single_phase_data():
    x = np.arange(11)
    retx, rety = _build_single_phase_data(x)
    np.testing.assert_array_equal(x, retx)
    np.testing.assert_array_equal(np.arange(1, 12) * np.pi, rety)


def test_build_single_phase_data_cb():
    x = np.arange(11)
    retx, rety = _build_single_phase_data(x, SPP_callbacks=[0])
    np.testing.assert_array_equal(np.arange(1, 11), retx)
    np.testing.assert_array_equal(np.arange(1, 11) * -np.pi, rety)


def test_build_single_phase_data_cb2():
    x = np.arange(11)
    retx, rety = _build_single_phase_data(x, SPP_callbacks=[0, 10])
    np.testing.assert_array_equal(np.arange(1, 10), retx)
    np.testing.assert_array_equal(np.arange(1, 10) * -np.pi, rety)


def test_build_single_phase_data_cb3():
    x = np.arange(11)
    retx, rety = _build_single_phase_data(x, SPP_callbacks=[2, 6])
    np.testing.assert_array_equal(np.array([0, 1, 3, 4, 5, 7, 8, 9, 10]), retx)
    np.testing.assert_array_almost_equal(
        np.array([3.141593, 6.283185, 3.141593, 0., -3.141593, 0., 3.141593, 6.283185, 9.424778]), rety
    )


def test_minmax_fail():
    with pytest.raises(TypeError):
        min_max_method(1, 2, 3, 4, SPP_callbacks="sda")


def test_minmax_basic():
    x, y = min_max_method(
        np.arange(10),
        np.arange(10),
        [],
        [],
        ref_point=0,
        minx=np.arange(0, 10)[::2],
        maxx=np.arange(0, 10)[1::2]
    )

    np.testing.assert_array_equal(np.arange(10), x)
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
        y
    )
