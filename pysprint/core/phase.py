"""
This file is not finished by any means.
"""
import warnings
from math import factorial

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

try:
    from lmfit import Model

    _has_lmfit = True
except ImportError:
    _has_lmfit = False

from pysprint.core.functions import _fit_config
from pysprint.utils import (
    print_disp,
    transform_lmfit_params_to_dispersion,
    transform_cf_params_to_dispersion,
    unpack_lmfit,
)


class Phase:
    """
    A class that represents a phase obtained from various
    methods.
    """

    is_dispersion_array = False
    is_coeff = False

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.poly = None
        self.fit = None

    def __call__(self, value):
        if self.poly:
            return self.poly.__call__(value)
        raise NotImplementedError(
            "Before calling, a polinomial must be fitted."
        )

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
    def from_coeff(cls, GD, GDD=0, TOD=0, FOD=0, QOD=0, SOD=0, domain=None):
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
        raise NotImplementedError(
            "Before calling, a polinomial must be fitted."
        )

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt
        if not self.is_dispersion_array or not self.is_coeff:
            ax.plot(self.x, self.y, **kwargs)
            if self.fit is not None:
                ax.plot(self.x, self.fit, "r--")
        else:
            ax.plot(self.x, self.poly(self.x), **kwargs)
        plt.grid()
        plt.show()

    @print_disp
    def fit(self, reference_point, order):
        if self.is_coeff or self.is_dispersion_array:
            warnings.warn("No need to fit another curve.")
            return

        self.fitorder = order

        _function = _fit_config[order]

        x, y = np.copy(self.x), np.copy(self.y)
        x -= reference_point

        if _has_lmfit:

            fitmodel = Model(_function)
            pars = fitmodel.make_params(
                **{f"b{i}": 1 for i in range(order + 1)}
            )
            result = fitmodel.fit(y, x=x, params=pars)
        else:
            popt, pcov = curve_fit(_function, x, y, maxfev=8000)

        if _has_lmfit:
            dispersion, dispersion_std = transform_lmfit_params_to_dispersion(
                *unpack_lmfit(result.params.items()), drop_first=True, dof=1
            )
            fit_report = result.fit_report()
            self.fit = result.best_fit()
        else:
            dispersion, dispersion_std = transform_cf_params_to_dispersion(
                popt, drop_first=True, dof=1
            )
            fit_report = (
                "To display detailed results,"
                " you must have `lmfit` installed."
            )

            self.fit = _function(x, *popt)

        return dispersion, dispersion_std, fit_report

    @property
    def order(self):
        if self.is_coeff or self.is_dispersion_array:
            self._order = self.poly.order
        else:
            self._order = self.fitorder
        return self._order

    @property
    def data(self):
        return self.x, self.y
