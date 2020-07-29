import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt

from pysprint.utils import calc_envelope
from pysprint.mpl_tools.peak import SelectButton


class DraggableEnvelope:

    # max absolute pixel distance to count as a hit
    epsilon = 5

    def __init__(self, x, y, mode="l"):
        plt.ion()
        rcParams["toolbar"] = "toolmanager"
        self.fig, self.ax = plt.subplots()
        self.x = x
        self.y = y
        self.mode = mode
        if self.mode == "l":
            self.envelope, self.y_env, self.loc = calc_envelope(
                self.y, np.arange(len(self.y)), "l"
            )
            plt.title("Adjust the lower envelope.")
        elif self.mode == "u":
            self.envelope, self.y_env, self.loc = calc_envelope(
                self.y, np.arange(len(self.y)), "u"
            )
            plt.title("Adjust the upper envelope.")
        else:
            raise ValueError("mode must be u or l.")
        # the active point index
        self._ind = None
        self.x_env = self.x[self.loc]

        (self.basedata,) = self.ax.plot(self.x, self.y)
        (self.lines,) = self.ax.plot(self.x, self.envelope, "r")
        (self.peakplot,) = self.ax.plot(self.x_env, self.y_env, "ko")

        self.fig.canvas.mpl_connect("button_press_event", self.button_press_callback)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_callback)
        self.fig.canvas.mpl_connect("draw_event", self.draw_callback)
        self.fig.canvas.mpl_connect(
            "button_release_event", self.button_release_callback
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)

        tm = self.fig.canvas.manager.toolmanager
        tm.add_tool("Toggle recording", SelectButton)
        self.fig.canvas.manager.toolbar.add_tool(
            tm.get_tool("Toggle recording"), "toolgroup"
        )
        self.my_select_button = tm.get_tool("Toggle recording")
        plt.grid()
        plt.show(block=True)

    def button_release_callback(self, event):
        """whenever a mouse button is released"""
        if event.button != 1:
            return
        self._ind = None

    def get_ind_under_point(self, event):
        """
        Get the index of the selected point within the given epsilon tolerance.
        """

        # We use the pixel coordinates, because the axes are usually really
        # differently scaled.
        x, y = self.peakplot.get_data()
        xy_pixels = self.ax.transData.transform(np.vstack([x, y]).T)
        xpix, ypix = xy_pixels.T

        # return the index of the point iff within epsilon distance.
        d = np.hypot(xpix - event.x, ypix - event.y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def button_press_callback(self, event):
        """whenever a mouse button is pressed we get the index"""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def key_press_callback(self, event):
        """whenever a key is pressed"""
        if not event.inaxes:
            return
        if self.my_select_button.toggled:
            if event.key == "d":
                ind = self.get_ind_under_point(event)
                if ind is not None:
                    self.x_env = np.delete(self.x_env, ind)
                    self.y_env = np.delete(self.y_env, ind)
                    self._interpolate()
                    self.peakplot.set_data(self.x_env, self.y_env)
                    self.lines.set_data(self.x, self.envelope)
            elif event.key == "i":
                self.y_env = np.append(self.y_env, event.ydata)
                self.x_env = np.append(self.x_env, event.xdata)
                self._interpolate()
                self.peakplot.set_data(self.x_env, self.y_env)
                self.lines.set_data(self.x, self.envelope)
            if self.peakplot.stale:
                self.fig.canvas.draw_idle()

    def _interpolate(self):
        """Upon modifying the datapoints we need to sort values and
        recalculate the interpolation."""
        idx = np.argsort(self.x_env)
        self.y_env, self.x_env = self.y_env[idx], self.x_env[idx]
        self.envelope = np.interp(self.x, self.x_env, self.y_env)

    def get_data(self):
        """
        Only returns the y values accordingly
        """
        if self.mode == "l":
            return self.y - self.envelope
        elif self.mode == "u":
            return self.y / self.envelope

    def draw_callback(self, event):
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.peakplot)
        self.ax.draw_artist(self.lines)

    def motion_notify_callback(self, event):
        """on mouse movement we move the selected point"""
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.my_select_button.toggled:
            x, y = event.xdata, event.ydata
            self.x_env[self._ind], self.y_env[self._ind] = x, y
            self._interpolate()
            self.peakplot.set_data(self.x_env, self.y_env)
            self.lines.set_data(self.x, self.envelope)

            # TODO: make faster drawing with background
            # self.fig.canvas.restore_region(self.background)

            self.ax.draw_artist(self.lines)
            self.ax.draw_artist(self.peakplot)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.draw()
