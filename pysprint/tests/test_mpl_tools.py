import os
from unittest import mock

import pytest
import numpy as np
from matplotlib import rcParams

from pysprint.mpl_tools.peak import EditPeak
from pysprint.mpl_tools.spp_editor import SPPEditor
from pysprint.mpl_tools.normalize import DraggableEnvelope


rcParams['figure.max_open_warning'] = 30


def mock_event(xdata, ydata, button, key, fig, canvas, inaxes=True):
    event = mock.Mock()
    event.button = button
    event.key = key
    event.xdata, event.ydata = xdata, ydata
    event.x, event.y = xdata, ydata
    event.inaxes = inaxes
    event.fig = fig
    event.canvas = canvas
    event.guiEvent = None
    event.name = 'MockEvent'

    return event


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


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_lock2(mock_show):
    x, y = np.arange(100), np.arange(100)
    xx, yy = np.array([4, 5]), np.array([4, 5])
    obj = EditPeak(x, y, x_extremal=xx, y_extremal=yy)
    obj.release()
    mck = mock_event(xdata=50, ydata=50, button="d", key="d", fig=obj.figure, canvas=obj.figure.canvas, inaxes=None)
    obj.on_clicked(event=mck)
    a, b = obj.get_dat
    np.testing.assert_array_equal(a, xx)
    np.testing.assert_array_equal(b, yy)
    mock_show.assert_called()


def test_inconsistent_length():
    with pytest.raises(ValueError):
        EditPeak(range(500), range(500), x_extremal=[1], y_extremal=[2, 3])


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_btn_release(mock_show):
    x = np.linspace(0, 6, 10000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="u")
    assert obj._ind is None
    y_transform = obj.get_data()
    np.testing.assert_allclose(y_transform[100:9900], np.ones(9800), atol=1, rtol=1)
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_keypress_cb(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="u")
    mck = mock_event(xdata=50, ydata=50, button="d", key="d", fig=obj.fig, canvas=obj.fig.canvas, inaxes=True)
    obj.key_press_callback(event=mck)
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_keypress_cb2(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="l")
    mck = mock_event(xdata=50, ydata=50, button="i", key="i", fig=obj.fig, canvas=obj.fig.canvas, inaxes=True)
    obj.key_press_callback(event=mck)
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_keypress_cb3(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    with pytest.raises(ValueError):
        DraggableEnvelope(x, y, mode="invalid")


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_button_press_cb(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="l")
    mck = mock_event(xdata=50, ydata=50, button="i", key="i", fig=obj.fig, canvas=obj.fig.canvas, inaxes=None)
    obj.button_press_callback(event=mck)
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_k_press_cb2(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="l")
    obj.epsilon = 1000000
    obj.x_env = np.array([56, 5])
    obj.y_env = np.array([60, 0.5])
    xy_pixels = obj.ax.transData.transform([5, 0.5])
    xpix, ypix = xy_pixels
    mck = mock_event(xdata=xpix, ydata=ypix, button="d", key="d", fig=obj.fig, canvas=obj.fig.canvas, inaxes=obj.ax)
    obj.key_press_callback(event=mck)
    assert 5 not in obj.x_env
    assert 0.5 not in obj.y_env
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_keypress_cb4(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="l")
    mck = mock_event(xdata=2, ydata=0.5, button="i", key="i", fig=obj.fig, canvas=obj.fig.canvas, inaxes=True)
    obj.key_press_callback(event=mck)
    assert 2 in obj.x_env
    assert 0.5 in obj.y_env
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_normalize_motion_notify_cb(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = DraggableEnvelope(x, y, mode="l")
    mck = mock_event(xdata=5, ydata=0.5, button="i", key="i", fig=obj.fig, canvas=obj.fig.canvas, inaxes=obj.ax)
    obj.motion_notify_callback(event=mck)
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_sppeditor_submit(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = SPPEditor(x, y)
    obj.submit("dsa50.dsa4")
    assert obj.delay == 50.4


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_sppeditor_text_change(mock_show):
    x = np.linspace(0, 6, 1000)
    y = np.cos(x)
    obj = SPPEditor(x, y)
    obj.text_change("dsa50.dsa4")
    assert obj.delay == 50.4


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_sppeditor_btn_release(mock_show):
    x = np.linspace(0, 6, 10000)
    y = np.cos(x)
    obj = SPPEditor(x, y)
    mck = mock_event(xdata=50, ydata=50, button=1, key=1, fig=obj.fig, canvas=obj.fig.canvas, inaxes=None)
    obj.button_release_callback(event=mck)
    assert obj._ind is None
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_sppeditor_get_ind(mock_show):
    x = np.linspace(0, 6, 10000)
    y = np.cos(x)
    obj = SPPEditor(x, y)
    obj.x_pos = np.array([1, 2])
    obj.y_pos = np.array([1, 2])
    mck = mock_event(xdata=1.5, ydata=1.3, button=1, key=1, fig=obj.fig, canvas=obj.fig.canvas, inaxes=obj.ax)
    obj.get_ind_under_point(event=mck)
    mock_show.assert_called()


@pytest.mark.skipif("TF_BUILD" in os.environ, reason="Azure fails this.")
@mock.patch("matplotlib.pyplot.show")
def test_sppeditor_keypress_cb(mock_show):
    x = np.linspace(0, 6, 10000)
    y = np.cos(x)
    obj = SPPEditor(x, y)
    obj.x_pos = np.array([1, 2])
    obj.y_pos = np.array([1, 2])
    mck = mock_event(xdata=2, ydata=0, button="i", key="i", fig=obj.fig, canvas=obj.fig.canvas, inaxes=obj.ax)
    obj.key_press_callback(event=mck)
    np.testing.assert_array_equal(obj.x_pos, np.array([1, 2, 2]))
    np.testing.assert_array_equal(obj.y_pos, np.array([1, 2, 0]))
    mock_show.assert_called()


@pytest.mark.skip(reason="Index can't be set..")
@mock.patch("matplotlib.pyplot.show")
def test_sppeditor_keypress_cb2(mock_show):
    x = np.linspace(0, 6, 10000)
    y = np.cos(x)
    obj = SPPEditor(x, y)
    obj.epsilon = 10000
    obj.x_pos = np.array([1, 2, 3])
    obj.y_pos = np.array([1, 2, 3])

    xy_pixels = obj.ax.transData.transform([2, 2])
    xpix, ypix = xy_pixels
    mck = mock_event(xdata=xpix, ydata=ypix, button="d", key="d", fig=obj.fig, canvas=obj.fig.canvas, inaxes=True)
    obj.key_press_callback(event=mck)
    np.testing.assert_array_equal(obj.x_pos, np.array([2]))
    np.testing.assert_array_equal(obj.y_pos, np.array([2]))
    mock_show.assert_called()
