from unittest.mock import patch
import threading

import pytest
import numpy as np

from pysprint.core.optimizer import FitOptimizer
from pysprint import Generator, CosFitMethod


@pytest.mark.skipif(
    not isinstance(threading.current_thread(), threading._MainThread),
    reason="threading breaks matplotlib.."
)
def test_optimizer1():
    x = np.arange(100)
    y = np.sin(x)

    opt = FitOptimizer(x, y, [], [], 50)

    with pytest.raises(ValueError):
        opt.run(r_extend_by=0.1, r_threshold=0.8)


@pytest.mark.skipif(
    not isinstance(threading.current_thread(), threading._MainThread),
    reason="threading breaks matplotlib.."
)
@pytest.mark.parametrize("val", [-10, 500])
def test_optimizer2(val):
    x = np.arange(100)
    y = np.sin(x)

    opt = FitOptimizer(x, y, [], [], 50)
    with pytest.raises(ValueError):
        opt.set_initial_region(val)


@pytest.mark.skipif(
    not isinstance(threading.current_thread(), threading._MainThread),
    reason="threading breaks matplotlib.."
)
def test_optimizer3():
    x = np.arange(100)
    y = np.sin(x)

    opt = FitOptimizer(x, y, [], [], 50)
    opt.set_final_guess(GD=1, GDD=50, TOD=36, FOD=48, QOD=None)
    assert len(opt.rest) == 3
    np.testing.assert_array_almost_equal(opt.rest, [25, 6, 2])
    np.testing.assert_array_almost_equal(opt.user_guess, [25, 6, 2])


# threading somehow breaks it, will be disabled until I figure it out
@pytest.mark.skipif(
    not isinstance(threading.current_thread(), threading._MainThread),
    reason="threading breaks matplotlib.."
)
@pytest.mark.parametrize("GD", [100, -500])
@pytest.mark.parametrize("GDD", [100, -500])
@pytest.mark.parametrize("delay", [0, 400])
def test_optimizer_from_api(delay, GD, GDD):
    g = Generator(1, 3, 2, delay, GD=GD, GDD=GDD, resolution=0.05, normalize=True)
    g.generate_freq()

    cf = CosFitMethod(*g.data)
    cf.guess_GD(GD + delay)
    cf.guess_GDD(GDD)
    with patch.object(FitOptimizer, "update_plot") as patched_obj:
        cf.optimizer(2, order=2, initial_region_ratio=0.01, extend_by=0.01)
        patched_obj.assert_called()


@pytest.mark.skipif(
    not isinstance(threading.current_thread(), threading._MainThread),
    reason="threading breaks matplotlib.."
)
def test_optimizer_from_api2():
    """
    Here we test that granting only GD value will yield correct results.
    """
    g = Generator(
        1,
        3,
        2,
        500,
        GD=400,
        GDD=400,
        TOD=800,
        FOD=7000,
        QOD=70000,
        resolution=0.05,
        normalize=True,
    )
    g.generate_freq()

    cf = CosFitMethod(*g.data)
    cf.guess_GD(900)
    with patch.object(FitOptimizer, "update_plot") as patched_obj:
        res = cf.optimizer(2, order=5, initial_region_ratio=0.01, extend_by=0.01)
        patched_obj.assert_called()
        np.testing.assert_array_almost_equal(res, [900, 400, 800, 7000, 70000])


@pytest.mark.parametrize("delay", list(range(100, 501, 100)))
def test_lookup(delay):
    g = Generator(1, 3, 2, delay, GD=0, GDD=500, TOD=800, normalize=True)
    g.generate_freq()
    cf = CosFitMethod(*g.data)
    cf.GD_lookup(2, engine="normal", silent=True)
    assert delay * 0.95 <= cf.params[3] <= delay * 1.05


def test_lookup2():
    g = Generator(1, 3, 2, 400, GD=0, GDD=500, normalize=False)
    g.generate_freq()
    cf = CosFitMethod(*g.data)
    cf.GD_lookup(200, engine="normal", silent=True)
    assert cf.params[3] == 1
