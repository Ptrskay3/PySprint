from math import factorial

import numpy as np

from pysprint.core.phase import Phase
from pysprint.config import _get_config_value
from pysprint.core.bases.dataset import Dataset
from pysprint.core._optimizer import FitOptimizer
from pysprint.core._evaluate import cff_method
from pysprint.utils import pprint_math_or_default
from pysprint.utils import NotCalculatedException
from pysprint.utils import pad_with_trailing_zeros

__all__ = ["CosFitMethod"]


class CosFitMethod(Dataset):
    """
    Basic interface for the Cosine Function Fit Method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.fit = None
        self.mt = 8000
        self.f = None
        self.r_squared = None
        self.phase = None

    def set_max_tries(self, value):
        """
        Overwrite the default scipy maximum try setting to fit the curve.
        """
        self.mt = value

    def adjust_offset(self, value):
        """
        Initial guess for offset.
        """
        self.params[0] = value

    def adjust_amplitude(self, value):
        """
        Initial guess for amplitude.
        """
        self.params[1] = value

    def guess_GD(self, value):
        """
        Initial guess for GD in fs.
        """
        self.params[3] = value

    def guess_GDD(self, value):
        """
        Initial guess for GDD in fs^2.
        """
        self.params[4] = value / 2

    def guess_TOD(self, value):
        """
        Initial guess for TOD in fs^3.
        """
        self.params[5] = value / 6

    def guess_FOD(self, value):
        """
        Initial guess for FOD in fs^4.
        """
        self.params[6] = value / 24

    def guess_QOD(self, value):
        """
        Initial guess for QOD in fs^5.
        """
        self.params[7] = value / 120

    def guess_SOD(self, value):
        """
        Initial guess for SOD in fs^6.
        """
        self.params[8] = value / 720

    def set_max_order(self, order):
        """
        Sets the maximum order of dispersion to look for.
        Should be called after guessing the initial parameters.

        Parameters
        ----------
        order : int
            Maximum order of dispersion to look for. Must be in [1, 6].
        """
        try:
            int(order)
        except ValueError as err:
            raise TypeError(
                "Order should be an in integer from [1, 6]."
            ) from err
        if order > 6 or order < 1:
            raise ValueError(
                "Order should be an in integer from [1, 6]."
            )
        order = 7 - order
        for i in range(1, order):
            self.params[-i] = 0

    def calculate(self, reference_point):
        """
        Cosine fit's calculate function.

        Parameters
        ----------
        reference_point: float
            Reference point on x axis.

        Returns
        -------
        dispersion : array-like
            [GD, GDD, TOD, FOD, QOD, SOD]
        dispersion_std : array-like
            Standard deviations due to uncertainty of the fit.
            They are only calculated if lmfit is installed.
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std, SOD_std]
        fit_report : str
            Not implemented yet. It returns an empty string for the time being.

        Note
        ----
        Decorated with pprint_disp, so the results are
        immediately printed without explicitly saying so.
        """
        dispersion, self.fit = cff_method(
            self.x,
            self.y,
            self.ref,
            self.sam,
            ref_point=reference_point,
            p0=self.params,
            maxtries=self.mt,
        )
        precision = _get_config_value("precision")
        self.r_squared = self._get_r_squared()
        pprint_math_or_default(f"R^2 = {self.r_squared:.{precision}f}\n")
        dispersion = pad_with_trailing_zeros(dispersion, 6)

        # TODO: This should produce the same result, but it does not.

        self.phase = Phase.from_dispersion_array(dispersion, domain=self.x)
        # trigger a fit to have disp calculated
        self.phase._fit(reference_point, order=np.max(np.flatnonzero(self.params)) - 2)
        return (
            dispersion,
            [0, 0, 0, 0, 0, 0],
            "",
        )

    def _get_r_squared(self):
        if self.fit is None:
            raise NotCalculatedException(
                "Must fit a curve before requesting r squared."
            )
        residuals = self.y_norm - self.fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.y_norm - np.mean(self.y_norm)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_result(self):
        """
        If the curve fitting is done, draws the fitted curve on the original
        dataset. Also prints the coeffitient of determination of the
        fit (a.k.a. r^2).
        """
        precision = _get_config_value("precision")
        try:
            self._get_r_squared()
            pprint_math_or_default(f"R^2 = {self.r_squared:.{precision}f}")
        except NotCalculatedException as e:
            raise ValueError("There's nothing to plot.") from e
        if self.fit is not None:
            self.plt.plot(self.x, self.fit, "k--", label="fit", zorder=99)
            self.plt.legend()
            self.plot()
            self.show()
        else:
            self.plot()
            self.show()

    def optimizer(
        self,
        reference_point,
        order=3,
        initial_region_ratio=0.1,
        extend_by=0.1,
        coef_threshold=0.3,
        max_tries=5000,
        show_endpoint=True,
    ):
        """
        Cosine fit optimizer. It's based on adding new terms to fit
        function successively until we reach the maximum order.

        Parameters
        ----------
        reference_point : float
            The reference point on the x axis.
        order : int, optional
            Polynomial (and maximum dispersion) order to fit. Must be in [1, 6].
            Default is 3.
        initial_region_ratio : float, optional
            The initial region in portion of the length of the dataset
            (0.1 will mean 10%, and so on..). Note that the bigger
            resolution the interferogram is, the lower it should be set.
            Default is 0.1. It should not be set too low, because too small
            initial region can result in failure.
        extend_by : float, optional
            The coefficient determining how quickly the region of fit is
            growing. The bigger resolution the interferogram is (or in general
            the higher the dispersion is), the lower it should be set.
            Default is 0.1.
        coef_threshold: float, optional
            The desired R^2 threshold which determines when to expand the region
            of fitting. It's often enough to leave it as is, however if you decide to
            change it, it is highly advised not to set a higher value than 0.7.
            Default is 0.3.
        max_tries : int, optional
            The maximum number of tries to fit a curve before failing.
            Default is 5000.
        show_endpoint : bool, optional
            If True show the fitting results when finished.
            Default is True.

        Note
        ----
        If the fit fails some parameters must be tweaked in order to
        achieve results. There is a list below with issues,
        its suspected reasons and solutions.

        **SciPy raises OptimizeWarning and the affected area is small
        or not showing any fit**

        Reasons:
        Completely wrong initial GD guess (or lack of guessing).
        Too broad inital region, so that the optimizer cannot find a
        suitable fit.

        This usually happens when the used data is large, or the spectral
        resolution is high.

        Solution:
        Provide better inital guess for GD.
        Lower the inital_region_ratio.

        **SciPy raises OptimizeWarning and the affected area is bigger**

        Reasons: When the optimizer steps up with order it also extends the
        region of fit.

        This error usually present when the region of fit is too quickly
        growing.

        Solution:
        Lower extend_by argument.

        **The optimizer is finished, but wrong fit is produced.**

        Reasons:
        We measure the goodness of fit with r^2 value. To allow this
        optimizer to smoothly find appropriate fits even for noisy datasets
        it's a good practice to keep the r^2 a lower value, such as
        the default 0.3. The way it works is we step up in order of fit
        (until max order) and extend region every time when a fit reaches
        the specified r^2 threshold value. This can be controlled via the
        coef_threshold argument.

        Solution:
        Adjust the coef_threshold value. Note that it's highly
        recommended not to set a higher value than 0.6.
        """
        x, y, ref, sam = self._safe_cast()
        self.f = FitOptimizer(
            x, y, ref, sam, reference_point=reference_point, max_order=order
        )
        self.f.set_initial_region(initial_region_ratio)
        self.f.set_final_guess(
            GD=self.params[3],
            GDD=self.params[4],
            TOD=self.params[5],
            FOD=self.params[6],
            QOD=self.params[7],
            SOD=self.params[8]
        )  # we can pass it higher params safely, they are ignored.
        disp = self.f.run(
            extend_by, coef_threshold, max_tries=max_tries, show_endpoint=show_endpoint,
        )
        disp = disp[3:]
        retval = [disp[i] * factorial(i + 1) for i in range(len(disp))]

        # TODO: This should produce the same result, but it does not.

        self.phase = Phase.from_dispersion_array(retval, domain=self.x)
        # trigger a fit to have disp calculated
        self.phase._fit(reference_point, order=order)
        self._dispersion_array = retval
        return retval
