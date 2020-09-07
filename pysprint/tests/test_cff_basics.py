from unittest.mock import patch

import pytest
import numpy as np

from pysprint import CosFitMethod


@pytest.fixture()
def make_obj():
    return CosFitMethod(np.arange(100), np.arange(100))


def test_offs(make_obj):
    c = make_obj
    c.adjust_offset(10)
    assert c.params[0] == 10


def test_amp(make_obj):
    c = make_obj
    c.adjust_amplitude(10)
    assert c.params[1] == 10


def test_disp_guesses(make_obj):
    c = make_obj
    c.guess_GD(10)
    c.guess_GDD(20)
    c.guess_TOD(30)
    c.guess_FOD(40)
    c.guess_QOD(50)
    c.guess_SOD(60)
    assert c.params[3] == 10
    assert c.params[4] == 20 / 2
    assert c.params[5] == 30 / 6
    assert c.params[6] == 40 / 24
    assert c.params[7] == 50 / 120
    assert c.params[8] == 60 / 720
    c.set_max_order(1)
    assert c.params[3] == 10
    assert c.params[4] == 0
    assert c.params[5] == 0
    assert c.params[6] == 0
    assert c.params[7] == 0
    assert c.params[8] == 0
    with pytest.raises(TypeError):
        c.set_max_order("invalid_text")
    with pytest.raises(ValueError):
        c.set_max_order(50)
    c.calculate(20)
    with patch("matplotlib.pyplot.show"):
        c.plot_result()
