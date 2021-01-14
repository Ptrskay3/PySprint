import sys
import warnings
from inspect import isfunction

import numpy as np
import matplotlib.pyplot as plt

from pysprint.core.methods.fftmethod import FFTMethod
from pysprint.core.phase import Phase
from pysprint.core._fft_tools import find_roi
from pysprint.core._fft_tools import find_center
from pysprint.utils.decorators import inplacify
from pysprint.utils import NotCalculatedException
from pysprint.utils import PySprintWarning
from pysprint.utils.misc import find_nearest
from pysprint.core.window import GaussianWindow
from pysprint.core.window import WindowBase

try:
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar
    CAN_PARALLELIZE = True
except ImportError:
    CAN_PARALLELIZE = False

    def delayed(func=None, *args, **kwargs):
        if isfunction(func):
            return func


class WFTMethod(FFTMethod):
    """Basic interface for Windowed Fourier Transform Method.
    The `window_class` attribuite can be set up for custom windowing.
    """

    def __init__(self, *args, **kwargs):
        self.window_class = kwargs.pop("window_class", GaussianWindow)
        assert issubclass(self.window_class, WindowBase), "window_class must subclass pysprint.core.window.WindowBase"
        super().__init__(*args, **kwargs)
        self.window_seq = {}
        self.found_centers = {}
        self.GD = None
        self.cachedlen = 0
        self.X_cont = np.array([])
        self.Y_cont = np.array([])
        self.Z_cont = np.array([])
        self.fastmath = True
        self.errorcounter = 0

    @inplacify
    def add_window(self, center, **kwargs):
        """
        Add a Gaussian window to the interferogram.

        Parameters
        ----------
        center : float
            The center of the Gaussian window.
        kwargs : dict
            Keyword arguments to pass to the `window_class`.
        """
        window = self.window_class(self.x, center=center, **kwargs)
        self.window_seq[center] = window
        return self

    @property
    def windows(self):
        return self.window_seq

    @property
    def centers(self):
        return self.window_seq.keys()

    @inplacify
    def add_window_generic(self, array, **kwargs):
        """
        Build a window sequence of given parameters with centers
        specified with ``array`` argument.

        Parameters
        ----------
        array : list, np.ndarray
            The array containing the centers of windows.
        kwargs : dict
            Keyword arguments to pass to the `window_class`.
        """
        if not isinstance(array, (list, np.ndarray)):
            raise TypeError("Expected list-like as ``array``.")
        for center in array:
            self.add_window(center=center, **kwargs)
        return self

    @inplacify
    def add_window_arange(self, start, stop, step, **kwargs):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.arange.

        Parameters
        ----------
        start : float
            The start of the centers.
        stop : float
            The end value of the center
        step : float
            The step value to increment center.
        kwargs : dict
            Keyword arguments to pass to the `window_class`.
        """
        arr = np.arange(start, stop, step)
        for cent in arr:
            self.add_window(center=cent, **kwargs)
        return self

    @inplacify
    def add_window_linspace(self, start, stop, num, **kwargs):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.linspace.

        Parameters
        ----------
        start : float
            The start of the centers.
        stop : float
            The end value of the center
        num : float
            The number of Gaussian windows.
        kwargs : dict
            Keyword arguments to pass to the `window_class`.
        """
        arr = np.linspace(start, stop, num)
        for cent in arr:
            self.add_window(center=cent, **kwargs)
        return self

    @inplacify
    def add_window_geomspace(self, start, stop, num, **kwargs):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.geomspace.

        Parameters
        ----------
        start : float
            The start of the centers.
        stop : float
            The end value of the center
        num : float
            The number of Gaussian windows.
        kwargs : dict
            Keyword arguments to pass to the `window_class`.
        """
        arr = np.geomspace(start, stop, num)
        for cent in arr:
            self.add_window(center=cent, **kwargs)
        return self

    def view_windows(self, ax=None, maxsize=80, **kwargs):
        """
        Gives a rough view of the different windows along with the ifg.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An axis to draw the plot on. If not given, it will plot
            of the last used axis.
        maxsize : int, optional
            The maximum number of Gaussian windows to display on plot.
            Default is 80, but be aware that setting a high value can
            drastically reduce performance.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot function.
        """
        winlen = len(self.window_seq)
        if maxsize != 0:
            ratio = winlen // maxsize
            if winlen > maxsize:
                warnings.warn(
                    "Image seems crowded, displaying only a subsample of the given windows.",
                    PySprintWarning
                )
                for i, (_, val) in enumerate(self.window_seq.items()):
                    if i % ratio == 0:
                        val.plot(ax=ax, scalefactor=np.max(self.y) * .75, **kwargs)
            else:
                for _, val in self.window_seq.items():
                    val.plot(ax=ax, scalefactor=np.max(self.y) * .75, **kwargs)
        self.plot(ax=ax)

    @inplacify
    def remove_all_windows(self):
        """
        Remove all the Gaussian windows.
        """
        self.window_seq.clear()
        return self

    @inplacify
    def reset_state(self):
        """
        Reset the object's state fully: delete all the
        calculated GD, caches, heatmaps and window sequences.
        """
        self.remove_all_windows()
        self.found_centers.clear()
        self.X_cont = np.array([])
        self.Y_cont = np.array([])
        self.Z_cont = np.array([])
        self.GD = None
        self.cachedlen = 0
        self.fastmath = True
        return self

    @inplacify
    def remove_window_at(self, center):
        """
        Removes a window at center.

        Parameters
        ----------
        center : float
            The center of the window to remove.
            Raises ValueError if there is not such window.
        """
        if center not in self.window_seq.keys():
            c = find_nearest(
                np.fromiter(self.window_seq.keys(), dtype=float), center
            )
            raise ValueError(
                f"There is no window with center {center}. "
                f"Did you mean {c[0]}?"
            )
        self.window_seq.pop(center, None)
        return self

    @inplacify
    def remove_window_interval(self, start, stop):
        """
        Remove window interval inclusively.

        Parameters
        ----------
        start : float
            The start value of the interval.
        stop : float
            The stop value of the interval.
        """
        wins = np.fromiter(self.window_seq.keys(), dtype=float)
        mask = wins[(wins <= stop) & (wins >= start)]
        for center in mask:
            self.window_seq.pop(center, None)
        return self

    @inplacify
    def cover(self, N, **kwargs):
        """
        Cover the whole domain with `N` number of windows
        uniformly built with the given parameters.

        Parameters
        ----------
        N : float
            The number of Gaussian windows.
        kwargs : dict
            Keyword arguments to pass to the `window_class`.
        """

        self.add_window_linspace(np.min(self.x), np.max(self.x), N, **kwargs)

    def calculate(
            self,
            reference_point,
            order,
            show_graph=False,
            silent=False,
            force_recalculate=False,
            fastmath=True,
            usenifft=False,
            parallel=False,
            ransac=False,
            errors="ignore",
            **kwds
    ):
        """
        Calculates the dispersion.

        Parameters
        ----------
        reference_point : float
            The reference point.
        order : int
            The dispersion order to look for. Must be in [2, 6].
        show_graph : bool, optional
            Whether to show the GD graph on complete. Default is False.
        silent : bool, optional
            Whether to print progressbar. By default it will print.
        force_recalculate : bool, optional
            Force to recalculate the GD graph not only the curve fitting.
            Default is False.
        fastmath : bool, optional
            Whether to build additional arrays to display heatmap.
            Default is True.
        usenifft : bool, optional
            Whether to use Non-unfirom FFT when calculating GD.
            Default is False. **Not stable.**
        parallel : bool, optional
            Whether to use parallel computation. Only availabe if `Dask`
            is installed. The speedup is about 50-70%. Default is False.
        ransac : bool, optional
            Whether to use RANSAC filtering on the detected peaks. Default
            is False.
        errors : str, optional
            Whether to raise an error is the algorithm couldn't find the
            center of the peak. Default is "ignore".
        kwds : optional
            Other keyword arguments to pass to RANSAC filter.

        Raises
        ------
        ValueError, if no window sequence is added to the interferogram.
        ValueError, if order is 1.
        ModuleNotFoundError, if `Dask` is not available when using parallel=True.
        """
        if len(self.window_seq) == 0:
            raise ValueError("Before calculating a window sequence must be set.")

        if self.cachedlen != len(self.window_seq) or fastmath != self.fastmath:
            force_recalculate = True
        self.fastmath = fastmath
        if force_recalculate:
            self.found_centers.clear()
            self.build_GD(
                silent=silent, fastmath=fastmath, usenifft=usenifft, parallel=parallel, errors=errors
            )
        if self.GD is None:
            self.build_GD(
                silent=silent, fastmath=fastmath, usenifft=usenifft, parallel=parallel, errors=errors
            )

        self.cachedlen = len(self.window_seq)

        if order == 1 or order > 6:
            raise ValueError("Order must be in [2, 6].")

        if ransac:
            print("Running RANSAC-filter..")
            self.GD.ransac_filter(order=order, plot=show_graph, **kwds)
            self.GD.apply_filter()

        d, ds, fr = self.GD._fit(
            reference_point=reference_point, order=order
        )
        if show_graph:
            self.GD.plot()
        return d, ds, fr

    def build_GD(self, silent=False, fastmath=True, usenifft=False, parallel=False, errors="ignore"):
        """
        Build the GD.

        Parameters
        ----------
        silent : bool, optional
            Whether to print progressbar. By default it will print.
        fastmath : bool, optional
            Whether to build additional arrays to display heatmap.
            Default is True.
        usenifft : bool, optional
            Whether to use Non-unfirom FFT when calculating GD.
            Default is False. **Not stable.**
        parallel : bool, optional
            Whether to use parallel computation. Only availabe if `Dask`
            is installed. The speedup is about 50-70%. Default is False.
        errors : str, optional
            Whether to raise an error is the algorithm couldn't find the
            center of the peak.

        Returns
        -------
        GD : pysprint.core.phase.Phase
            The phase object with `GD_mode=True`. See its docstring for more info.
        """
        if parallel:

            if not CAN_PARALLELIZE:
                raise ModuleNotFoundError(
                    "Module `dask` not found. Please install it in order to use parallelism."
                )

            else:
                self.fastmath = fastmath
                self._apply_window_seq_parallel(fastmath=fastmath, usenifft=usenifft, errors=errors)

                if not silent:
                    with ProgressBar():
                        computed = compute(*self.found_centers.values())

                else:
                    computed = compute(*self.found_centers.values())

                cleaned_delays = [
                    k for i, k in enumerate(self.found_centers.keys()) if computed[i] is not None
                ]
                delay = np.fromiter(cleaned_delays, dtype=float)

                omega = np.fromiter([c for c in computed if c is not None], dtype=float)

                if not silent:
                    print(f"Skipped: {len(self.window_seq) - sum(1 for _ in filter(None.__ne__, computed))}")

        else:
            self.fastmath = fastmath
            self._apply_window_sequence(silent=silent, fastmath=fastmath, usenifft=usenifft)
            self._clean_centers()

            delay = np.fromiter(self.found_centers.keys(), dtype=float)
            omega = np.fromiter(self.found_centers.values(), dtype=float)

        self.GD = Phase(delay, omega, GD_mode=True)
        return self.GD

    def build_phase(self):
        raise NotImplementedError("Use `build_GD` instead.")

    def _predict_ideal_window_fwhm(self):
        pass

    def _apply_window_sequence(
            self, silent=False, fastmath=True, usenifft=False, errors="ignore"
    ):
        winlen = len(self.window_seq)
        self.errorcounter = 0
        if not fastmath:
            # here we setup the shape for the Z array because
            # it is much faster than using np.append in every iteration
            _x, _y, _, _ = self._safe_cast()
            _obj = FFTMethod(_x, _y)
            _obj.ifft(usenifft=usenifft)
            x, y = find_roi(_obj.x, _obj.y)
            self.Y_cont = np.array(x)
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
                self.Z_cont[:, idx] = y
            try:
                centx, _ = find_center(x, y)
                self.found_centers[_center] = centx
            except ValueError as err:
                self.errorcounter += 1
                if errors == "ignore":
                    self.found_centers[_center] = None
                else:
                    raise err
            if not silent:  # This creates about 5-15% overhead.. maybe create a buffer
                sys.stdout.write('\r')
                j = (idx + 1) / winlen
                sys.stdout.write(
                    "Progress : [%-30s] %d%% (Skipped: %d)" % ('=' * int(30 * j), 100 * j, self.errorcounter)
                )
                sys.stdout.flush()

    def _apply_window_seq_parallel(
            self, fastmath=True, usenifft=False, errors="ignore"
    ):
        self.errorcounter = 0
        if not fastmath:
            # here we setup the shape for the Z array and allocate Y, because
            # it is much faster than using np.append in every iteration
            _x, _y, _, _ = self._safe_cast()
            _obj = FFTMethod(_x, _y)
            _obj.ifft(usenifft=usenifft)
            x, y = find_roi(_obj.x, _obj.y)
            yshape = y.size
            self.Y_cont = np.array(x)
            xshape = len(self.window_seq)
            self.Z_cont = np.empty(shape=(yshape, xshape))

        for idx, (_center, _window) in enumerate(self.window_seq.items()):
            element = self._prepare_element(idx, _window, fastmath, usenifft, errors)
            if element is None:
                self.errorcounter += 1  # This might be useless, since we lazy evaluate things..
            self.found_centers[_center] = element

    @delayed
    def _prepare_element(self, idx, window, fastmath=True, usenifft=False, errors="ignore"):
        _x, _y, _, _ = self._safe_cast()
        _obj = FFTMethod(_x, _y)
        _obj.y *= window.y
        _obj.ifft(usenifft=usenifft)
        x, y = find_roi(_obj.x, _obj.y)
        if not fastmath:
            self.Z_cont[:, idx] = y
        try:
            centx, _ = find_center(x, y)
            return centx
        except ValueError as err:
            if errors == "ignore":
                return None
            else:
                raise err

    def _clean_centers(self, silent=False):
        dct = {k: v for k, v in self.found_centers.items() if v is not None}
        self.found_centers = dct
        winlen = len(self.window_seq)
        usefullen = len(self.found_centers)

        if not silent:
            if winlen != usefullen:
                print(
                    f"\n{abs(winlen-usefullen)} points skipped "
                    f"due to ambiguous peak position."
                )

    def errorplot(self, *args, **kwargs):
        """
        Plot the errors of fitting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An axis to draw the plot on. If not given, it will plot
            of the last used axis.
        percent : bool, optional
            Whether to plot percentage difference. Default is False.
        title : str, optional
            The title of the plot. Default is "Errors".
        kwargs : dict, optional
            Additional keyword arguments to pass to plot function.
        """
        try:
            getattr(self.GD, "errorplot", None)(*args, **kwargs)
        except TypeError:
            raise NotCalculatedException("Must calculate before plotting errors.")

    @property
    def get_GD(self):
        """
        Return the GD if it is already calculated.
        """
        if self.GD is not None:
            return self.GD
        raise NotCalculatedException("Must calculate GD first.")

    @property
    def errors(self):
        """
        Return the fitting errors as np.ndarray.
        """
        return getattr(self.GD, "errors", None)

    def _collect_failures(self):
        return [k for k in self.window_seq.keys() if k not in self.found_centers.keys()]

    def _construct_heatmap_data(self):
        self.X_cont = np.fromiter(self.window_seq.keys(), dtype=float)

    def heatmap(self, ax=None, levels=None, cmap="viridis", include_ridge=True):
        """
        Plot the heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An axis to draw the plot on. If not given, it will plot
            of the last used axis.
        levels : np.ndarray, optional
            The levels to use for plotting.
        cmap : str, optional
            The colormap to use.
        include_ridge : bool, optional
            Whether to mark the detected ridge of the plot.
            Default is True.
        """
        if self.GD is None:
            raise NotCalculatedException("Must calculate GD first.")

        if self.fastmath:
            raise ValueError(
                "You need to recalculate with `fastmath=False` to plot the heatmap."
            )
        # Only construct if we need to..
        if not (self.Y_cont.size, self.X_cont.size) == self.Z_cont.shape:
            self._construct_heatmap_data()
        if ax is None:
            plt.contourf(
                self.X_cont, self.Y_cont, self.Z_cont, levels=levels, cmap=cmap, extend="both"
            )
        else:
            ax.contourf(
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
            try:
                upper_bound = min(1.5 * np.max(self.GD.data[1]), np.max(self.Y_cont))
                plt.ylim(None, upper_bound)
            except ValueError:
                pass
            # ValueError: zero-size array to reduction operation maximum which has no identity
            # This means that the array is empty, we should pass that case.
        else:
            ax.set_autoscalex_on(False)
            try:
                upper_bound = min(1.5 * np.max(self.GD.data[1]), np.max(self.Y_cont))
                ax.set(
                    xlabel="Window center [PHz]",
                    ylabel="Delay [fs]",
                    ylim=(None, upper_bound)
                )
            except ValueError:
                pass

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
