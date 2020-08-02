import pytest
import numpy as np

from pysprint.core.bases import Dataset
from pysprint.core.methods import FFTMethod
from pysprint.core.methods.generator import Generator


@pytest.fixture()
def setup_ifg():
    x = np.linspace(-1, 8, 1000)
    y = np.sin(x) * np.cos(x)
    noarm = Dataset(x, y)
    arm = Dataset(x, y, y - 1, y - 2)
    return noarm, arm


def test_axis(setup_ifg):
    a, _ = setup_ifg
    with pytest.raises(ValueError):
        a.transform(None, axis="dsa")


def test_apply_selffunc_noarm(setup_ifg):
    a, _ = setup_ifg
    before = a.x
    a.transform("chdomain", axis=0)
    a.transform("chdomain", axis=0)
    after = a.x

    np.testing.assert_array_almost_equal(before, after)


def test_apply_selffunc_special_axis_arg():

    x = np.linspace(-1, 8, 1000)
    y = np.sin(x) * np.cos(x)
    f = FFTMethod(x, y)
    before = f.x
    f.transform("shift", axis=0)
    f.transform("shift", axis=0)
    after = f.x
    np.testing.assert_array_almost_equal(before, after)


def test_apply_selffunc_arm(setup_ifg):
    _, a = setup_ifg
    before = a.x
    a.transform("chdomain", axis=0)
    a.transform("chdomain", axis=0)
    after = a.x

    np.testing.assert_array_almost_equal(before, after)


def test_apply_np_ufunc_noarm(setup_ifg):
    a, _ = setup_ifg
    before = a.y_norm
    a.transform(np.sqrt, axis=1)
    after = a.y_norm

    np.testing.assert_array_almost_equal(np.sqrt(before), after)


def test_apply_functype(setup_ifg):
    a, _ = setup_ifg

    def f(y, n, k=3):
        if n < y < k:
            return 1
        else:
            return 0

    a.transform(f, args=(0,), kwargs={"k": 0.5}, axis=1)
    assert np.all(a.y[np.where(a.y > 0)][0] == 1)
    assert np.all(a.y[np.where(a.y < 0.5)][0] == 0)


def test_apply_functype2(setup_ifg):
    _, a = setup_ifg

    def f(y, n, k=3):
        if n < y < k:
            return 1
        else:
            return 0

    a.transform(f, args=(0,), kwargs={"k": 0.5}, axis=1)
    assert np.all(a.y[np.where(a.y > 0)][0] == 1)
    assert np.all(a.y[np.where(a.y < 0.5)][0] == 0)


def test_validation(setup_ifg):
    a, _ = setup_ifg

    def splitup(x):
        return -2
    with pytest.raises(ValueError):
        a.transform(splitup, axis=0)


def test_validation2(setup_ifg):
    a, _ = setup_ifg

    @np.vectorize
    def csum(x):
        return np.reshape(x, newshape=(2, -1))
    a.transform(csum, axis=0)
