import warnings
import logging
from math import factorial
from collections import namedtuple

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from pysprint.config import _get_config_value

try:
    from lmfit import Model

    _has_lmfit = True
except ImportError:
    _has_lmfit = False

from pysprint.core._functions import _fit_config
from pysprint.core._preprocess import cut_data
from pysprint.utils import (
    pprint_disp,
    transform_lmfit_params_to_dispersion,
    transform_cf_params_to_dispersion,
    _unpack_lmfit,
    find_nearest,
    inplacify,
    NotCalculatedException,
)

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

        # Make coeffs available after fitting
        self.coef_temp = namedtuple('coef_temp', ['GD', 'GDD', 'TOD', 'FOD', 'QOD', 'SOD'])
        self.coef_array = None

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
    def from_disperion_array(cls, dispersion_array, domain=None):
        cls.is_dispersion_array = True
        if domain is None:
            x = np.linspace(2, 4, num=2000)
        else:
            x = np.asarray(domain)
        coeffs = [i / factorial(i + 1) for i in dispersion_array]
        cls.poly = np.poly1d(coeffs[::-1])
        return cls(x, cls.poly(x))

    @classmethod
    def from_coeff(cls, GD=0, GDD=0, TOD=0, FOD=0, QOD=0, SOD=0, domain=None):
        if domain is None:
            x = np.linspace(2, 4, num=2000)
        else:
            x = np.asarray(domain)

        cls.is_coeff = True
        cls.poly = np.poly1d([SOD, QOD, FOD, TOD, GDD, GD])
        return cls(x, cls.poly(x))

    def __str__(self):
        if self.poly is not None:
            return self.poly.__str__()
        raise NotImplementedError("Before calling, a polynomial must be fitted.")

    def plot(self, ax=None, **kwargs):
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
            ax.plot(x, y, **kwargs)
            if self.fitted_curve is not None:
                ax.plot(x, self.fitted_curve[idx], "r--")
            else:
                # ax.plot(x, self.poly(x)[idx], **kwargs)
                pass

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
                dispersion, dispersion_std = transform_cf_params_to_dispersion(
                    popt, drop_first=True, dof=1
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
            residuals = self.y - self.fitted_curve
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
            return 1 - (ss_res / ss_tot)
        raise NotCalculatedException

    @property
    def r_squared(self):
        """
        The value of R^2.
        """
        return self._get_r_squared()

    # TODO
    def del_range(self, start, stop):
        pass
