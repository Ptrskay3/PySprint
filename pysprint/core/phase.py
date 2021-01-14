import os
import ast
import errno
from collections import namedtuple
import logging
import warnings
from math import factorial

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

try:
    from lmfit import Model

    _has_lmfit = True
except ImportError:
    _has_lmfit = False

from pysprint.config import _get_config_value
from pysprint.core._functions import _fit_config
from pysprint.core._preprocess import cut_data
from pysprint.core.ransac import run_regressor
from pysprint.utils import pprint_disp
from pysprint.utils import transform_lmfit_params_to_dispersion
from pysprint.utils import transform_cf_params_to_dispersion
from pysprint.utils import find_nearest
from pysprint.utils import inplacify
from pysprint.utils import NotCalculatedException
from pysprint.utils.misc import _unpack_lmfit

logger = logging.getLogger(__name__)
FORMAT = "[ %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)


class Phase:
    """
    A class that represents a phase obtained from various
    methods.
    """

    def __init__(self, x, y, GD_mode=False):
        """
        Phase constructor.

        Parameters
        ----------
        x : np.ndarray
            The domain of the phase.
        y : np.ndarray
            The y values of the phase.
        GD_mode : bool, optional
            Whether to treat dataset as a GD graph
            instead of plain phase. Default is False.
        """
        self.x = x
        self.y = y
        self.poly = None
        self.fitted_curve = None
        self.is_dispersion_array = False
        self.is_coeff = False
        self.fitorder = None
        self.GD_mode = GD_mode
        self._filtered_x = None
        self._filtered_y = None

        # Make coeffs available after fitting
        self.coef_temp = namedtuple(
            'coef_temp', ['GD', 'GDD', 'TOD', 'FOD', 'QOD', 'SOD']
        )
        self.coef_std_temp = namedtuple(
            'coef_std_temp', ['GD_err', 'GDD_err', 'TOD_err', 'FOD_err', 'QOD_err', 'SOD_err']
        )

        self.coef_array = None
        self.coef_std_array = None

    @classmethod
    def from_log(cls, filename, **kwargs):
        return cls(*[_to_array(x) for x in _lastlines(filename, 2)])

    @property
    def GD(self):
        if self.coef_array is not None:
            return self.coef_array.GD

    @property
    def GDD(self):
        if self.coef_array is not None:
            return self.coef_array.GDD

    @property
    def TOD(self):
        if self.coef_array is not None:
            return self.coef_array.TOD

    @property
    def FOD(self):
        if self.coef_array is not None:
            return self.coef_array.FOD

    @property
    def QOD(self):
        if self.coef_array is not None:
            return self.coef_array.QOD

    @property
    def SOD(self):
        if self.coef_array is not None:
            return self.coef_array.SOD

    @property
    def GD_err(self):
        if self.coef_std_array is not None:
            return self.coef_std_array.GD_err

    @property
    def GDD_err(self):
        if self.coef_std_array is not None:
            return self.coef_std_array.GDD_err

    @property
    def TOD_err(self):
        if self.coef_std_array is not None:
            return self.coef_std_array.TOD_err

    @property
    def FOD_err(self):
        if self.coef_std_array is not None:
            return self.coef_std_array.FOD_err

    @property
    def QOD_err(self):
        if self.coef_std_array is not None:
            return self.coef_std_array.QOD_err

    @property
    def SOD_err(self):
        if self.coef_std_array is not None:
            return self.coef_std_array.SOD_err

    def __call__(self, value):
        if self.poly:
            return self.poly.__call__(value)
        raise NotImplementedError("Before calling, a polinomial must be fitted.")

    @inplacify
    def slice(self, start=None, stop=None):
        """
        Cuts the dataset on x axis.

        Parameters
        ----------
        start : float
            start value of cutting interval
            Not giving a value will keep the dataset's original minimum value.
            Note that giving `None` will leave original minimum untouched too.
            Default is `None`.
        stop : float
            stop value of cutting interval
            Not giving a value will keep the dataset's original maximum value.
            Note that giving `None` will leave original maximum untouched too.
            Default is `None`.
        """
        self.x, self.y = cut_data(self.x, self.y, [], [], start=start, stop=stop)
        return self

    @classmethod
    def from_dispersion_array(cls, dispersion_array, domain=None):
        cls.is_dispersion_array = True
        # print(dispersion_array)
        if domain is None:
            x = np.linspace(2, 4, num=2000)
        else:
            x = np.asarray(domain)
        coeffs = [0] + [v / factorial(i + 1) for i, v in enumerate(dispersion_array)]
        # print(coeffs)
        cls.poly = np.poly1d(coeffs[::-1])
        return cls(x, cls.poly(x))

    @classmethod
    def from_coeff(cls, GD=0, GDD=0, TOD=0, FOD=0, QOD=0, SOD=0, domain=None):
        if domain is None:
            x = np.linspace(2, 4, num=2000)
        else:
            x = np.asarray(domain)

        cls.is_coeff = True
        cls.poly = np.poly1d([SOD, QOD, FOD, TOD, GDD, GD, 0])
        return cls(x, cls.poly(x))

    def __str__(self):
        if self.poly is not None:
            return self.poly.__str__()
        return super().__str__()

    def plot(self, ax=None, marker=None, linestyle=None, **kwargs):
        """
        Plot the phase and the fitted curve (if there's any).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An axis to draw the plot on. If not given, it will plot
            of the last used axis.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot function.
        """
        if ax is None:
            ax = plt
            plt.xlabel(r"$\omega\,[PHz]$")
        else:
            ax.set(xlabel=r"$\omega\,[PHz]$")
        if not self.is_dispersion_array or not self.is_coeff:
            # we need to sort them because plots become messy
            # if we keep it unsorted
            idx = np.argsort(self.x)
            x, y = self.x[idx], self.y[idx]
            marker = marker or kwargs.pop("marker", ".")
            linestyle = linestyle or kwargs.pop("linestyle", "None")
            ax.plot(x, y, marker=marker, linestyle=linestyle, **kwargs)
            if self.fitted_curve is not None:
                ax.plot(x, self.fitted_curve[idx], "r--", label="Current fit")

    @pprint_disp
    def fit(self, reference_point, order):
        """
        Fit the phase and determine dispersion coefficients.

        Parameters
        ----------

        reference_point : float
            The reference point to use for fitting.
        order : int
            The order of dispersion to look for. Must be in [1, 6].
        """
        return self._fit(reference_point=reference_point, order=order)

    def _fit(self, reference_point, order):
        """
        This is meant to be used privately, when the pprint_disp
        is handled by another function. The `fit` method is for
        public use.
        """
        if self.is_coeff or self.is_dispersion_array:
            warnings.warn("No need to fit another curve.")
            return
        else:
            if self.GD_mode:
                order -= 1
            self.fitorder = order

            _function = _fit_config[order]

            x, y = np.copy(self.x), np.copy(self.y)
            x -= reference_point
            # idx = np.argsort(x)
            # x, y = x[idx], y[idx]

            if _has_lmfit:

                fitmodel = Model(_function)
                pars = fitmodel.make_params(**{f"b{i}": 1 for i in range(order + 1)})
                result = fitmodel.fit(y, x=x, params=pars)
            else:
                popt, pcov = curve_fit(_function, x, y, maxfev=8000)

            if _has_lmfit:
                dispersion, dispersion_std = transform_lmfit_params_to_dispersion(
                    *_unpack_lmfit(result.params.items()), drop_first=True, dof=1
                )
                fit_report = result.fit_report()
                self.fitted_curve = result.best_fit
            else:
                # IMPORTANT! We must take a copy of `popt`, because that's modified
                # inplace. This caused wrong plots when `lmfit` wasn't installed.
                # version fixed: 0.13.2
                dispersion, dispersion_std = transform_cf_params_to_dispersion(
                    np.copy(popt), drop_first=True, dof=1
                )
                fit_report = (
                    "To display detailed results you must have `lmfit` installed."
                )
                self.fitted_curve = _function(x, *popt)

            if self.GD_mode:
                _, idx = find_nearest(self.x, reference_point)
                dispersion = np.insert(dispersion, 0, self.fitted_curve[idx])
                dispersion_std = np.insert(dispersion_std, 0, 0)

                # The below line must have 7 elements, so slice these redundant coeffs..
                dispersion, dispersion_std = dispersion[:-1], dispersion_std[:-1]

            self.coef_array = self.coef_temp(*dispersion)
            self.coef_std_array = self.coef_std_temp(*dispersion_std)

            return dispersion, dispersion_std, fit_report

    def errorplot(self, ax=None, percent=False, title="Errors", **kwargs):
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
        if ax is None:
            ax = plt.gca()
            plt.xlabel("$\omega\, [PHz]$")
            if percent:
                plt.ylabel("%")
            plt.title(title)
        else:
            ax.set(xlabel="$\omega\, [PHz]$", title=title)
            if percent:
                ax.set(ylabel="%")

        idx = np.argsort(self.x)
        x, y = self.x[idx], self.y[idx]
        if self.fitted_curve is not None:
            if percent:
                ax.plot(x, np.abs((y - self.fitted_curve[idx]) / y) * 100, **kwargs)
            else:
                ax.plot(x, y - self.fitted_curve[idx], **kwargs)
            ax.text(
                0.02,
                0.95,
                f"$R^2 = {self._get_r_squared():.{_get_config_value('precision')}f}$",
                transform=ax.transAxes,
                fontsize="medium",
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                zorder=100
            )
            ax.grid()
        else:
            raise NotCalculatedException("Must fit a curve before requesting errors.")

    def flip_around(self, value, side="left"):
        """
        Flip the phase's y values.

        Parameters
        ----------
        value : float
            The x value where to perform flipping.
        side : str, optional
            The side where to flip the y values. Default is "left".
        """
        if side == "left":
            idx = np.where(self.x <= value)[0]
        elif side == "right":
            idx = np.where(self.x >= value)[0]
        else:
            idx = np.array([])

        x_to_flip, y_to_flip = self.x[idx], self.y[idx]

        _, cls_idx = find_nearest(x_to_flip, value)
        logger.info(f"Using {self.x[cls_idx]} instead {value} as flip center.")
        value_base = self.y[cls_idx]
        y_to_flip *= -1
        y_to_flip += 2 * value_base
        self.y[idx] = y_to_flip

    @property
    def errors(self):
        """
        Return the fitting errors as np.ndarray.
        """
        if self.fitted_curve is not None:
            return self.y - self.fitted_curve
        raise ValueError("Must fit a curve before requesting errors.")

    @property
    def order(self):
        """
        Return the order of polynomial which was fitted.
        """
        if self.is_coeff or self.is_dispersion_array:
            self._order = self.poly.order
        else:
            self._order = self.fitorder
        return self._order

    @property
    def dispersion_order(self):
        """
        Return the dispersion order.
        """
        return self.fitorder + 1 if self.GD_mode else self.fitorder

    # TODO : For consistency, this should return a pd.DataFrame.
    # Because we hardwired this into other functions it's not safe
    # to rewrite, but this definitely should be corrected in the future.
    @property
    def data(self):
        """
        Return the data as tuple (x, y).
        """
        return self.x, self.y

    def _get_r_squared(self):
        """
        Calculate the R^2 value.
        """
        if self.fitted_curve is not None:
            try:
                residuals = self.y - self.fitted_curve
            except ValueError as e:
                msg = ValueError("Original data was modified, cannot broadcast shape with the fitted curve.")
                raise msg from e
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
            if ss_res == 0 or ss_tot == 0:
                warnings.warn("R squared could not be computed properly.")
            return 1 - (ss_res / ss_tot)
        raise NotCalculatedException

    @property
    def r_squared(self):
        """
        The value of R^2.
        """
        return self._get_r_squared()

    @inplacify
    def remove_range(self, start=None, stop=None):
        """
        Remove a part of the phase for x that satisfies
        start <= x <= stop. Leaving start (stop) as `None`
        will remove from the beginning (end).
        """
        if start is None:
            start = np.min(self.x)
        if stop is None:
            stop = np.max(self.x)

        mask = ((self.x <= stop) & (self.x >= start))
        self.x, self.y = self.x[~mask], self.y[~mask]
        return self

    @inplacify
    def smooth(self, method="gaussian_filter1d", sigma=10, **kwargs):
        """
        Smooth the phase with a piecewise cubic polynomial
        which is twice continuously differentiable.
        """
        if method not in ("savitzky-golay", "convolve", "medfilt", "gaussian_filter1d"):
            raise ValueError(
                "method must be 'savitzky-golay', 'medfilt', 'gaussian_filter1d' or 'convolve'"
            )

        window_length = kwargs.pop("window_lenth", 51)

        if method == "savitzky-golay":
            poly_order = kwargs.pop("poly_order", 3)
            self.y = savgol_filter(self.y, window_length, poly_order, **kwargs)

        elif method == "convolve":
            box = np.ones(window_length) / window_length
            self.y = np.convolve(self.y, box, mode=kwargs.get("mode", "same"))

        elif method == "medfilt":
            self.y = medfilt(self.y, kwargs.get("kernel_size", 5))

        elif method == "gaussian_filter1d":
            sigma = sigma or kwargs.pop("sigma", 10)
            self.y = gaussian_filter1d(self.y, sigma=sigma, **kwargs)

        return self

    # TODO: Remove the duplicated logic. This function is in pysprint's init.py
    # and we can't circular import it. It should be moved to a separate file.
    def plot_outside(self, *args, **kwargs):
        """
        Plot the current dataset out of the notebook. For detailed
        parameters see `Dataset.plot` function.
        """
        backend = kwargs.pop("backend", "Qt5Agg")
        original_backend = plt.get_backend()
        try:
            plt.switch_backend(backend)
            self.plot(*args, **kwargs)
            plt.show(block=True)
        except (AttributeError, ImportError, ModuleNotFoundError) as err:
            raise ValueError(
                f"Couldn't set backend {backend}, you should manually "
                "change to an appropriate GUI backend."
            ) from err
        finally:
            plt.switch_backend(original_backend)

    def ransac_filter(self, order=None, plot=False, **kwds):
        """
        Perform a RANSAC (RANdom SAmple Consensus) filter to the dataset,
        which detects outliers. This function will *only* plot the results.
        To actually apply it, use the `apply_filter` method.

        Parameters
        ----------
        order : int, optional
            The degree of polynomial to estimate the shape of the curve.
            This argument must be given if no fitting was performed before.
        plot : bool, optional
            Whether to plot the result. Default is False.
        kwds : dict, optional
            Other arguments to pass to sklearn.linear_model.RANSACRegressor.
            The most important is `residual_threshold`, which measures how
            distant points should be filtered out.
        """
        if order is None and self.fitorder is not None:
            order = self.fitorder
        if order is None and self.fitorder is None:
            raise ValueError("Must specify fit order for RANSAC filtering.")
        self._filtered_x, self._filtered_y = run_regressor(
            self, degree=order, plot=plot, **kwds
        )

    @inplacify
    def apply_filter(self):
        """
        Apply the RANSAC filter.
        """
        if self._filtered_x is None or self._filtered_y is None:
            raise ValueError("There's nothing to apply.")
        prec = _get_config_value("precision")
        if len(self.x) != len(self._filtered_x):
            r = len(self.x) - len(self._filtered_x)
            print(f"Values dropped: {r} ({(r / len(self.x) * 100):.{prec}f} % of total)")
        self.x, self.y = self._filtered_x, self._filtered_y
        return self


def _lastlines(file, n, bsize=2048):
    '''
    Return the last `n` lines of the log file **efficiently**.
    https://stackoverflow.com/a/12295054/11751294
    '''
    # get newlines type, open in universal mode to find it
    with open(file, 'r') as hfile:
        if not hfile.readline():
            return  # empty, no point
        sep = hfile.newlines.encode()
#     assert isinstance(sep, str), 'multiple newline types found, aborting'

    # find a suitable seek position in binary mode
    with open(file, 'rb') as hfile:
        hfile.seek(0, os.SEEK_END)
        linecount = 0
        pos = 0

        while linecount <= n + 1:
            # read at least n lines + 1 more; we need to skip a partial line later on
            try:
                hfile.seek(-bsize, os.SEEK_CUR)           # go backwards
                linecount += hfile.read(bsize).count(sep) # count newlines
                hfile.seek(-bsize, os.SEEK_CUR)           # go back again
            except IOError as e:
                if e.errno == errno.EINVAL:
                    # Attempted to seek past the start, can't go further
                    bsize = hfile.tell()
                    hfile.seek(0, os.SEEK_SET)
                    pos = 0
                    linecount += hfile.read(bsize).count(sep)
                    break
                raise  # Some other I/O exception, re-raise
            pos = hfile.tell()

    # Re-open in text mode
    with open(file, 'r') as hfile:
        hfile.seek(pos, os.SEEK_SET)  # our file position from above

        for line in hfile:
            # We've located n lines *or more*, so skip if needed
            if linecount > n:
                linecount -= 1
                continue
            # The rest we yield
            yield line

def _to_array(x):
    y = x.split(': ')[-1].strip('\n')
    return np.array(ast.literal_eval(y))

