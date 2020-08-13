import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

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

        scalefactor : float, optional
            Number describing how much a given should be scaled up ONLY
            for visibility.

        zorder : float, optional
            The drawing order of artists is determined by their zorder attribute, which is
            a floating point number. Artists with higher zorder are drawn on top. You can
            change the order for individual artists by setting their zorder. The default
            value depends on the type of the Artist.
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
        self.secondary_centers = {}
        self.tertiary_centers = {}
        self.GD = None
        self.cachedlen = 0
        self.X_cont = np.array([])
        self.Y_cont = np.array([])
        self.Z_cont = np.array([])
        self.fastmath = True

    @mutually_exclusive_args("std", "fwhm")
    def add_window(self, center, std=None, fwhm=None, order=2):
        if not np.min(self.x) <= center <= np.max(self.x):
            raise ValueError(
                f"Cannot add window at {center}, because "
                f"it is out of the dataset's range (from {np.min(self.x):.3f} to {np.max(self.x):.3f})."
            )
        if std:
            window = Window.from_std(
                self.x, center=center, std=std, order=order
            )
        else:
            window = Window(self.x, center=center, fwhm=fwhm, order=order)
        self.window_seq[center] = window

    @property
    def windows(self):
        return self.window_seq

    @property
    def centers(self):
        return self.window_seq.keys()

    @mutually_exclusive_args("std", "fwhm")
    def add_window_arange(
        self, start, stop, step, std=None, fwhm=None, order=2
    ):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.arange.
        """
        arr = np.arange(start, stop, step)
        for cent in arr:
            if std:
                self.add_window(center=cent, std=std, order=order)
            else:
                self.add_window(center=cent, fwhm=fwhm, order=order)

    @mutually_exclusive_args("std", "fwhm")
    def add_window_linspace(
        self, start, stop, num, std=None, fwhm=None, order=2
    ):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.linspace.
        """
        arr = np.linspace(start, stop, num)
        for cent in arr:
            if std:
                self.add_window(center=cent, std=std, order=order)
            else:
                self.add_window(center=cent, fwhm=fwhm, order=order)

    @mutually_exclusive_args("std", "fwhm")
    def add_window_geomspace(
        self, start, stop, num, std=None, fwhm=None, order=2
    ):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.geomspace.
        """
        arr = np.geomspace(start, stop, num)
        for cent in arr:
            if std:
                self.add_window(center=cent, std=std, order=order)
            else:
                self.add_window(center=cent, fwhm=fwhm, order=order)

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
                    val.plot(ax=ax, scalefactor=np.max(self.y) * .75, **kwargs)
        else:
            for _, val in self.window_seq.items():
                val.plot(ax=ax, scalefactor=np.max(self.y) * .75, **kwargs)
        self.plot(ax=ax)

    def remove_all_windows(self):
        self.window_seq.clear()

    def reset_state(self):
        self.remove_all_windows()
        self.found_centers.clear()
        self.X_cont = np.array([])
        self.Y_cont = np.array([])
        self.Z_cont = np.array([])
        self.GD = None
        self.cachedlen = 0
        self.fastmath = True

    def remove_window_at(self, center):
        if center not in self.window_seq.keys():
            c = find_nearest(
                np.fromiter(self.window_seq.keys(), dtype=float), center
            )
            raise ValueError(
                f"There is no window with center {center}. "
                f"Did you mean {c[0]}?"
            )
        self.window_seq.pop(center, None)

    def remove_window_interval(self, start, stop):
        pass

    # TODO : Add parameter to describe how many peaks we are looking for..
    def calculate(
            self,
            reference_point,
            order,
            show_graph=False,
            silent=False,
            force_recalculate=False,
            fastmath=True,
            usenifft=False
    ):
        if len(self.window_seq) == 0:
            raise ValueError("Before calculating a window sequence must be set.")

        if self.cachedlen != len(self.window_seq) or fastmath != self.fastmath:
            force_recalculate = True
        self.fastmath = fastmath
        if force_recalculate:
            self.found_centers.clear()
            self.build_GD(silent=silent, fastmath=fastmath, usenifft=usenifft)
        if self.GD is None:
            self.build_GD(silent=silent, fastmath=fastmath, usenifft=usenifft)

        self.cachedlen = len(self.window_seq)

        if order == 1:
            raise ValueError("Cannot fit constant function to data. Order must be in [2, 5].")

        d, ds, fr = self.GD._fit(
            reference_point=reference_point, order=order
        )
        if show_graph:
            self.GD.plot()
        return d, ds, fr

    def build_GD(self, silent=False, fastmath=True, usenifft=False):
        self.fastmath = fastmath
        self._apply_window_sequence(silent=silent, fastmath=fastmath, usenifft=usenifft)
        self._clean_centers()
        delay = np.fromiter(self.found_centers.keys(), dtype=float)
        omega = np.fromiter(self.found_centers.values(), dtype=float)
        self.GD = Phase(delay, omega, GD_mode=True)
        return self.GD

    def _predict_ideal_window_fwhm(self):
        pass

    def _apply_window_sequence(
            self, silent=False, fastmath=True, usenifft=False, errors="ignore"
    ):
        winlen = len(self.window_seq)

        if not fastmath:
            # here we setup the shape for the Z array because
            # it is much faster than using np.append in every iteration
            _x, _y, _, _ = self._safe_cast()
            _obj = FFTMethod(_x, _y)
            _obj.ifft(usenifft=usenifft)
            x, y = find_roi(_obj.x, _obj.y)
            yshape = y.size
            xshape = len(self.window_seq)
            self.Z_cont = np.empty(shape=(yshape, xshape))

        for idx, (_center, _window) in enumerate(self.window_seq.items()):
            _x, _y, _, _ = self._safe_cast()
            _obj = FFTMethod(_x, _y)
            _obj.y *= _window.y
            _obj.ifft(usenifft=usenifft)
            x, y = find_roi(_obj.x, _obj.y)
            if not fastmath:
                if self.Y_cont.size == 0:  # prevent allocating it in every iteration
                    self.Y_cont = np.array(x)
                self.Z_cont[:, idx] = y
            try:
                centx, _ = find_center(x, y)
                self.found_centers[_center] = centx
            except ValueError as err:
                if errors == "ignore":
                    self.found_centers[_center] = None
                else:
                    raise err
            if not silent:
                sys.stdout.write('\r')
                j = (idx + 1) / winlen
                sys.stdout.write("Progress : [%-30s] %d%%" % ('=' * int(30 * j), 100 * j))
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

    def errorplot(self, *args, **kwargs):
        try:
            getattr(self.GD, "errorplot", None)(*args, **kwargs)
        except TypeError:
            raise ValueError("Must calculate before plotting errors.")

    @property
    def get_GD(self):
        if self.GD is not None:
            return self.GD
        raise ValueError("Must calculate GD first.")

    @property
    def errors(self):
        return getattr(self.GD, "errors", None)

    # TODO : Integrate this into _apply_window_sequence and add matplotlib dispatcher
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
            _obj.plt.plot(xx, yy, markersize=10, marker="*")
        except ValueError:
            pass
        _obj.plot()
        _obj.show()

    def _construct_heatmap_data(self):
        self.X_cont = np.fromiter(self.window_seq.keys(), dtype=float)

    def heatmap(self, ax=None, levels=None, cmap="viridis", include_ridge=True):
        if self.GD is None:
            raise ValueError("Must calculate first.")

        if self.fastmath:
            raise ValueError(
                "You need to recalculate with `fastmath=False` to plot the heatmap."
            )
        # Only construct if we need to..
        if not (self.Y_cont.size, self.X_cont.size) == self.Z_cont.shape:
            self._construct_heatmap_data()
        if levels is None:
            levels = np.linspace(0, 0.02, 30)
        else:
            if not isinstance(levels, np.ndarray):
                raise ValueError("Expected np.ndarray as levels.")
        if ax is None:
            cs = plt.contourf(
                self.X_cont, self.Y_cont, self.Z_cont, levels=levels, cmap=cmap, extend="both"
            )
        else:
            cs = ax.contourf(
                self.X_cont, self.Y_cont, self.Z_cont, levels=levels, cmap=cmap, extend="both"
            )
        if include_ridge:
            if ax is None:
                plt.plot(*self.GD.data, color='red', label='detected ridge')
            else:
                ax.plot(*self.GD.data, color='red', label='detected ridge')
            plt.legend()
        if ax is None:
            plt.xlabel('Window center [PHz]')
            plt.ylabel('Delay [fs]')
            plt.ylim(None, 1.5 * np.max(self.GD.data[1]))
        else:
            ax.set_autoscalex_on(False)
            ax.set(
                xlabel="Window center [PHz]",
                ylabel="Delay [fs]",
                ylim=(None, 1.5 * np.max(self.GD.data[1]))
            )

    def get_heatmap_data(self):
        """
        Return the data which was used to create the heatmap.

        Returns
        -------

        X_cont : np.ndarray
            The window centers with shape (n,).

        Y_cont : np.ndarray
            The time axis calculated from the IFFT of the dataset with shape (m,).

        Z_cont : np.ndarray
            2D array with shape (m, n) containing the depth information.
        """
        if all([self.Y_cont.size != 0, self.Z_cont.size != 0]):
            self._construct_heatmap_data()
        else:
            raise ValueError("Must calculate with `fastmath=False` before trying to access the heatmap data.")

        return self.X_cont, self.Y_cont, self.Z_cont
