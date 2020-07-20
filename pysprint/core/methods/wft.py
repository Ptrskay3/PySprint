import sys
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
    windows with caching the y values.
    """

    def __init__(self, x, center, fwhm, order=2):
        self.x = x
        self.center = center
        self.fwhm = fwhm
        self.order = order

    @lazy_property
    def y(self):
        return gaussian_window(self.x, self.center, self.fwhm, self.order)

    @classmethod
    def from_std(cls, x, center, std, order=2):
        _fwhm = std * 2 * np.log(2) ** (1 / order)
        return cls(x, center, _fwhm, order)

    def __repr__(self):
        return f"Window(center={self.center:.5f}, fwhm={self.fwhm}, order={self.order})"

    def plot(self, ax=None, scalefactor=1, zorder=90, **kwargs):
        """
        Plot the window.

        Parameters
        ----------

        ax : axis
            The axis to plot on. If not given, plot on the last axis.
        """
        if ax is None:
            ax = plt
        ax.plot(self.x, self.y * scalefactor, zorder=zorder, **kwargs)


class WFTMethod(FFTMethod):
    """Basic interface for Windowed Fourier Transform Method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_seq = {}
        self.found_centers = {}
        self.GD = None
        self.cachedlen = 0
        self.X_cont = np.array([])
        self.Y_cont = np.array([])
        self.Z_cont = np.array([])
        self.fastmath = True

    @mutually_exclusive_args("std", "fwhm")
    def add_window(self, center, std=None, fwhm=None, order=2, **kwargs):
        if not np.min(self.x) <= center <= np.max(self.x):
            raise ValueError(
                f"Cannot add window at {center}, because "
                f"it is out of the dataset's range (from {np.min(self.x):.3f} to {np.max(self.x):.3f})."
            )
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

    @property
    def centers(self):
        return self.window_seq.keys()

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

    #  TODO : scale them up for visibility
    def view_windows(self, ax=None, maxsize=80, **kwargs):
        """
        Gives a rough view of the different windows along with the ifg.
        """
        winlen = len(self.window_seq)
        ratio = winlen % maxsize
        if winlen > maxsize:
            warnings.warn(
                "Image seems crowded, displaying only a sequence of the given windows.",
                PySprintWarning
            )
            for i, (_, val) in enumerate(self.window_seq.items()):
                if i % ratio == 0:
                    val.plot(ax=ax, scalefactor=np.max(self.y)*.75, **kwargs)
        else:
            for _, val in self.window_seq.items():
                val.plot(ax=ax, scalefactor=np.max(self.y)*.75, **kwargs)
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

    # TODO : Add parameter to describe how many peaks we are looking for..
    def calculate(
            self, reference_point, order, show_graph=False, silent=False, force_recalculate=False, fastmath=True
    ):

        if self.cachedlen != len(self.window_seq) or fastmath != self.fastmath:
            force_recalculate = True
        self.fastmath = fastmath
        if force_recalculate:
            self.found_centers.clear()
            self.retrieve_GD(silent=silent, fastmath=fastmath)
        if self.GD is None:
            self.retrieve_GD(silent=silent, fastmath=fastmath)

        self.cachedlen = len(self.window_seq)

        d, ds, fr = self.GD._fit(
            reference_point=reference_point, order=order - 1
        )
        if show_graph:
            self.GD.plot()

        delay = np.fromiter(self.found_centers.keys(), dtype=float)
        val, _ = find_nearest(delay, reference_point)
        GD_val = self.found_centers[val]
        d = np.insert(d, 0, GD_val)  # manually insert the GD value
        ds = np.insert(ds, 0, 0)     # because we obtain the GD curve this way.
        return d, ds, fr

    def retrieve_GD(self, silent, fastmath=True):
        self._apply_window_sequence(silent=silent, fastmath=fastmath)
        self._clean_centers()
        delay = np.fromiter(self.found_centers.keys(), dtype=float)
        omega = np.fromiter(self.found_centers.values(), dtype=float)
        self.GD = Phase(delay, omega)
        return self.GD

    def _predict_ideal_window_fwhm(self):
        pass

    def _apply_window_sequence(self, silent=False, fastmath=True):
        winlen = len(self.window_seq)
        for idx, (_center, _window) in enumerate(self.window_seq.items()):
            _x, _y, _, _ = self._safe_cast()
            _obj = FFTMethod(_x, _y)
            _obj.y *= _window.y
            _obj.ifft()
            x, y = find_roi(_obj.x, _obj.y)
            if not fastmath:
                self.Y_cont = np.array(x)
                self.Z_cont = np.append(self.Z_cont, y)
            try:
                centx, _ = find_center(x, y)
                self.found_centers[_center] = centx
            except ValueError:
                self.found_centers[_center] = None
            if not silent:
                sys.stdout.write('\r')
                j = (idx + 1) / winlen
                sys.stdout.write("Progress : [%-30s] %d%%" % ('='*int(30*j), 100*j))
                sys.stdout.flush()

    def _clean_centers(self, silent=False):
        dct = {k: v for k, v in self.found_centers.items() if v is not None}
        self.found_centers = dct
        winlen = len(self.window_seq)
        usefullen = len(self.found_centers)

        if not silent:
            if winlen != usefullen:
                print(
                    f"\nIn total {winlen-usefullen} out of {winlen} datapoints "
                    f"were thrown away due to ambiguous peak positions."
                    )

    def _prepare_element(self, center):
        if center not in self.window_seq.keys():
            raise ValueError(
                f"Window with center {center} cannot be found."
            )
        _x, _y, _, _ = self._safe_cast()
        _obj = FFTMethod(_x, _y)
        _obj.y *= self.window_seq[center].y
        _obj.ifft()
        x, y = find_roi(_obj.x, _obj.y)
        try:
            xx, yy = find_center(x, y)
            _obj.plotwidget.plot(xx, yy, markersize=10, marker="*")
        except ValueError:
            pass
        _obj.show()

    def heatmap(self, levels=None):
        if self.GD is None:
            raise ValueError

        if self.fastmath:
            raise ValueError(
                "You need to recalculate with `fastmath=False` to plot the heatmap."
            )
        self.X_cont = np.fromiter(self.window_seq.keys(), dtype=float)
        print(self.Z_cont.shape)
        self.Z_cont = np.reshape(self.Z_cont, (-1, len(self.X_cont)), order="F")
        print(self.Z_cont.shape)
        if levels is None:
            levels = np.linspace(0, 0.02, 50)
        else:
            assert isinstance(levels, np.ndarray)
        plt.figure(figsize=(10, 10))
        plt.contourf(self.X_cont, self.Y_cont, self.Z_cont, levels=levels)
        plt.colorbar()
        plt.plot(*self.GD.data, color='red', label='detected ridge')
        plt.xlabel('Window position [PHz]')
        plt.ylabel('Delay [fs]')
        plt.ylim(None, 1.5 * np.max(self.GD.data[1]))
        plt.legend()
        plt.show(block=True)


