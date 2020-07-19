"""
Experimental
"""
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox


class SPPEditor:

    epsilon = 7  # max absolute pixel distance to count as a hit

    def __init__(self, x, y):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.set_figheight(6)
        self.fig.set_figwidth(10)
        plt.grid()
        plt.subplots_adjust(bottom=0.2)
        self.x = x
        self.y = y
        self._ind = None  # the active point index
        self.x_pos, self.y_pos = np.array([]), np.array([])
        (self.basedata,) = self.ax.plot(self.x, self.y)
        (self.points,) = self.ax.plot(self.x_pos, self.y_pos, "ko")
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_callback)
        self.fig.canvas.mpl_connect(
            "button_release_event", self.button_release_callback
        )
        self.axbox = plt.axes([0.1, 0.05, 0.8, 0.1])
        self.text_box = TextBox(self.axbox, "Delay [fs]", initial="0")
        self.text_box.on_submit(self.submit)
        self.text_box.on_text_change(self.text_change)

        plt.show(block=True)

    def submit(self, delay):
        try:
            delay = re.sub(r"[^0-9\.,\-]", "", delay)
            self.delay = float(delay)
        except ValueError:
            pass  # we ignore bad calls

    def text_change(self, delay):
        try:
            delay = re.sub(r"[^0-9\.,\-]", "", delay)
            self.delay = float(delay)
        except ValueError:
            pass  # we ignore bad calls

    def get_data(self):
        positions, _ = self.points.get_data()
        if not hasattr(self, "delay"):
            self.delay = 0
        self.delay = np.ones_like(positions) * self.delay
        return self.delay, positions

    def button_release_callback(self, event):
        """whenever a mouse button is released"""
        if event.button != 1:
            return
        self._ind = None

    def get_ind_under_point(self, event):
        """
        Get the index of the selected point within the given epsilon tolerance
        """

        # We use the pixel coordinates, because the axes are usually really
        # differently scaled.
        if event.inaxes is None:
            return
        if event.inaxes in [self.ax]:
            try:
                x, y = self.points.get_data()
                xy_pixels = self.ax.transData.transform(np.vstack([x, y]).T)
                xpix, ypix = xy_pixels.T

                # return the index of the point iff within epsilon distance.
                d = np.hypot(xpix - event.x, ypix - event.y)
                (indseq,) = np.nonzero(d == d.min())
                ind = indseq[0]

                if d[ind] >= self.epsilon:
                    ind = None
            except ValueError:
                return
            return ind

    def key_press_callback(self, event):
        """whenever a key is pressed"""
        if not event.inaxes:
            return
        if event.key == "d":
            if event.inaxes in [self.ax]:
                ind = self.get_ind_under_point(event)
            else:
                ind = None
            if ind is not None:
                self.x_pos = np.delete(self.x_pos, ind)
                self.y_pos = np.delete(self.y_pos, ind)
                self.points.set_data(self.x_pos, self.y_pos)

        elif event.key == "i":
            if event.inaxes in [self.ax]:
                self.x_pos = np.append(self.x_pos, event.xdata)
                self.y_pos = np.append(self.y_pos, event.ydata)
                self.points.set_data(self.x_pos, self.y_pos)

        if self.points.stale:
            self.fig.canvas.draw_idle()
