import os
from unittest import mock
from contextlib import contextmanager

import pytest
import numpy as np
import matplotlib.pyplot as plt

from pysprint.mpl_tools.peak import EditPeak

def mock_event(xdata, ydata, button, key, fig, canvas, inaxes=True):

    event = mock.Mock()
    event.button = button
    event.key = key
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = inaxes
    event.fig = fig
    event.canvas = canvas
    event.guiEvent = None
    event.name = 'MockEvent'

    return event

import contextlib

@contextmanager
def temporal_setattr(obj, attr, new_value):
    replaced = False
    old_value = None
    if hasattr(obj, attr):
        if attr in obj.__dict__:
            replaced = True
        if replaced:
            old_value = getattr(obj, attr)
    setattr(obj, attr, new_value)
    yield replaced, old_value
    if not replaced:
        delattr(obj, attr)
    else:
        setattr(obj, attr, old_value)


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_insert(mock_show):
    x, y = np.arange(100), np.arange(100)
    xx, yy = np.array([4, 5]), np.array([4, 5])
    obj = EditPeak(x, y, x_extremal=xx, y_extremal=yy)
    mck = mock_event(xdata=50, ydata=50, button="i", key="i", fig=obj.figure, canvas=obj.figure.canvas, inaxes=True)
    obj.on_clicked(event=mck)
    a, b = obj.get_dat
    np.testing.assert_array_equal(a, np.array([4, 5, 50]))
    np.testing.assert_array_equal(b, np.array([4, 5, 50]))
    mock_show.assert_called()

@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_delete(mock_show):
    x, y = np.arange(100), np.arange(100)
    xx, yy = np.array([4, 5]), np.array([4, 5])
    obj = EditPeak(x, y, x_extremal=xx, y_extremal=yy)
    mck = mock_event(xdata=50, ydata=50, button="d", key="d", fig=obj.figure, canvas=obj.figure.canvas, inaxes=True)
    obj.on_clicked(event=mck)
    a, b = obj.get_dat
    np.testing.assert_array_equal(a, np.array([4]))
    np.testing.assert_array_equal(b, np.array([4]))
    mock_show.assert_called()

@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_not_inaxes(mock_show):
    x, y = np.arange(100), np.arange(100)
    xx, yy = np.array([4, 5]), np.array([4, 5])
    obj = EditPeak(x, y, x_extremal=xx, y_extremal=yy)
    mck = mock_event(xdata=50, ydata=50, button="d", key="d", fig=obj.figure, canvas=obj.figure.canvas, inaxes=None)
    obj.on_clicked(event=mck)
    a, b = obj.get_dat
    np.testing.assert_array_equal(a, xx)
    np.testing.assert_array_equal(b, yy)
    mock_show.assert_called()

@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_lock(mock_show):
    x, y = np.arange(100), np.arange(100)
    xx, yy = np.array([4, 5]), np.array([4, 5])
    obj = EditPeak(x, y, x_extremal=xx, y_extremal=yy)
    obj.my_select_button.disable()
    mck = mock_event(xdata=50, ydata=50, button="d", key="d", fig=obj.figure, canvas=obj.figure.canvas, inaxes=None)
    obj.on_clicked(event=mck)
    a, b = obj.get_dat
    np.testing.assert_array_equal(a, xx)
    np.testing.assert_array_equal(b, yy)
    mock_show.assert_called()