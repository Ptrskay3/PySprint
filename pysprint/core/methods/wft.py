import warnings

import numpy as np
import matplotlib.pyplot as plt # noqa

from pysprint.core.phase import Phase
from pysprint.utils import PySprintWarning
from pysprint.core.fft_tools import find_center, find_roi
from pysprint.core.methods._fft import FFTMethod
from pysprint.core.evaluate import gaussian_window
from pysprint.utils.misc import mutually_exclusive_args, lazy_property, find_nearest


class Window:
    """
    Basic class that implements functionality related to Gaussian
    windows.
    """

    def __init__(self, x, center, fwhm, order=2, **kwargs):
        self.x = x
        self.center = center
        self.fwhm = fwhm
        self.order = order
        self.mpl_style = kwargs

    @lazy_property
    def y(self):
        return gaussian_window(self.x, self.center, self.fwhm, self.order)

    @classmethod
    def from_std(cls, x, center, std, order=2):
        _fwhm = std * 2 * np.log(2) ** (1 / order)
        return cls(x, center, _fwhm, order)

    def plot(self, ax=None, zorder=90):
        """
        Plot the window.

        Parameters
        ----------

        ax : axis
            The axis to plot on. If not given, plot on the last axis.
        """
        if ax is None:
            ax = plt
        ax.plot(self.x, self.y, zorder=zorder, **self.mpl_style)


class WFTMethod(FFTMethod):
    """Basic interface for Windowed Fourier Transform Method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_seq = {}
        self.found_centers = {}
        self.phase = None

    @mutually_exclusive_args("std", "fwhm")
    def add_window(self, center, std=None, fwhm=None, order=2, **kwargs):
        if std:
            window = Window.from_std(
                self.x, center=center, std=std, order=order, **kwargs
            )
        else:
            window = Window(self.x, center=center, fwhm=fwhm, order=order, **kwargs)
        self.window_seq[center] = window

    @property
    def windows(self):
        return self.window_seq

    @mutually_exclusive_args("std", "fwhm")
    def add_window_arange(
        self, start, stop, step, std=None, fwhm=None, order=2, **kwargs
    ):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.arange.
        """
        arr = np.arange(start, stop, step)
        for cent in arr:
            if std:
                self.add_window(center=cent, std=std, order=order, **kwargs)
            else:
                self.add_window(center=cent, fwhm=fwhm, order=order, **kwargs)

    @mutually_exclusive_args("std", "fwhm")
    def add_window_linspace(
        self, start, stop, num, std=None, fwhm=None, order=2, **kwargs
    ):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.linspace.
        """
        arr = np.linspace(start, stop, num)
        for cent in arr:
            if std:
                self.add_window(center=cent, std=std, order=order, **kwargs)
            else:
                self.add_window(center=cent, fwhm=fwhm, order=order, **kwargs)

    @mutually_exclusive_args("std", "fwhm")
    def add_window_geomspace(
        self, start, stop, num, std=None, fwhm=None, order=2, **kwargs
    ):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.geomspace.
        """
        arr = np.geomspace(start, stop, num)
        for cent in arr:
            if std:
                self.add_window(center=cent, std=std, order=order, **kwargs)
            else:
                self.add_window(center=cent, fwhm=fwhm, order=order, **kwargs)

    def view_windows(self, ax=None, maxsize=80):
        """
        Gives a rough view of the different windows along with the ifg.
        """
        # TODO: check for too crowded image later and warn user.
        # Maybe just display a sample that case?
        winlen = len(self.window_seq)
        ratio = winlen % maxsize
        if winlen > maxsize:
            warnings.warn(
                "Too crowded image, displaying only a sequence", PySprintWarning
            )
            for i, (_, val) in enumerate(self.window_seq.items()):
                if i % ratio == 0:
                    val.plot(ax=ax)
        else:
            for _, val in self.window_seq.items():
                val.plot(ax=ax)
        self.show()

    def remove_all_windows(self):
        self.window_seq.clear()

    def remove_window_at(self, center):
        if center not in self.window_seq.keys():
            raise ValueError(
                f"Cannot remove window with center {center}, "
                f"because it cannot be found."
            )
        self.window_seq.pop(center, None)

    # TODO : Rewrite this accordingly
    def calculate(
            self, reference_point, order, show_graph=False, silent=False, force_recalculate=False
    ):
        if force_recalculate:
            self.found_centers.clear()
            self.retrieve_GD(silent=silent)
        if self.phase is None:
            self.retrieve_GD(silent=silent)

        d, ds, fr = self.phase._fit(
            reference_point=reference_point, order=order - 1
        )
        if show_graph:
            self.phase.plot()

        delay = np.fromiter(self.found_centers.keys(), dtype=float)
        val, _ = find_nearest(delay, reference_point)
        GD_val = self.found_centers[val]
        d = np.insert(d, 0, GD_val)
        ds = np.insert(ds, 0, 0)
        return d, ds, fr

    def retrieve_GD(self, silent):
        self._apply_window_sequence(silent=silent)
        delay = np.fromiter(self.found_centers.keys(), dtype=float)
        omega = np.fromiter(self.found_centers.values(), dtype=float)
        self.phase = Phase(delay, omega)
        return self.phase

    def _predict_ideal_window_fwhm(self):
        pass

    def _apply_window_sequence(self, silent=False):
        winlen = len(self.window_seq)
        for idx, (_center, _window) in enumerate(self.window_seq.items()):
            _x, _y, _, _ = self._safe_cast()
            _obj = FFTMethod(_x, _y)
            _obj.y *= _window.y
            _obj.ifft()
            x, y = find_roi(_obj.x, _obj.y)
            centx, _ = find_center(x, y)
            self.found_centers[_center] = centx
            currlen = len(self.found_centers)
            if not silent:
                if idx % 10 == 0:
                    print(f"Progress:{currlen}/{winlen}")
