from math import factorial

import numpy as np

from pysprint.core.bases.dataset import Dataset
from pysprint.core.optimizer import FitOptimizer
from pysprint.core.evaluate import cff_method


__all__ = ["CosFitMethod"]


class CosFitMethod(Dataset):
    """
    Basic interface for the Cosine Function Fit Method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = [1, 1, 1, 1, 1, 1, 1, 1]
        self.fit = None
        self.mt = 8000
        self.f = None

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

    def set_max_order(self, order):
        """
        Sets the maximum order of dispersion to look for.
        Should be called after guessing the initial parameters.

        Parameters:
        ----------

        order : int
            Maximum order of dispersion to look for. Must be in [1, 5].
        """
        if order > 5 or order < 1:
            print("Order should be an in integer from [1, 5].")
        try:
            int(order)
        except ValueError:
            print("Order should be an in integer from [1, 5].")
        order = 6 - order
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
        dispersion: array-like
            [GD, GDD, TOD, FOD, QOD]

        dispersion_std: array-like
            Standard deviations due to uncertainty of the fit.
            They are only calculated if lmfit is installed.
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

        fit_report: string
            Not implemented yet. It returns an empty string for the time being.

        Notes:
        ------

        Decorated with print_disp, so the results are
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
        dispersion = list(dispersion)
        while len(dispersion) < 5:
            dispersion.append(0)
        return (
            dispersion,
            [0, 0, 0, 0, 0],
            "",
        )

    def plot_result(self):
        """
        If the curve fitting is done, draws the fitted curve on the original
        dataset. Also prints the coeffitient of determination of the
        fit (a.k.a. r^2).
        """
        try:
            residuals = self.y_norm - self.fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((self.y_norm - np.mean(self.y_norm)) ** 2)
            print("r^2 = " + str(1 - (ss_res / ss_tot)))
        except Exception:  # TODO: handle that blank exception
            pass
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
            Polynomial (and maximum dispersion) order to fit. Must be in [1, 5].
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

        Notes
        -----
        If the fit fails some parameters must be tweaked in order to
        achieve results. There is a list below with issues,
        its suspected reasons and solutions.

        **SciPy raises OptimizeWarning and the affected area is small
        or not showing any fit

        Reasons:
        - Completely wrong initial GD guess (or lack of guessing).
        - Too broad inital region, so that the optimizer cannot find a
        suitable fit.

        This usually happens when the used data is large, or the spectral
        resolution is high.

        Solution:
        - Provide better inital guess for GD.
        - Lower the inital_region_ratio.

        **SciPy raises OptimizeWarning and the affected area is bigger

        Reasons:
        - When the optimizer steps up with order it also extends the
        region of fit.

        This error usually present when the region of fit is too quickly
        growing.

        Solution:
        - Lower extend_by argument.

        **The optimizer is finished, but wrong fit is produced.

        Reasons:
        - We measure the goodness of fit with r^2 value. To allow this
        optimizer to smoothly find appropriate fits even for noisy datasets
        it's a good practice to keep the r^2 a lower value, such as
        the default 0.3. The way it works is we step up in order of fit
        (until max order) and extend region every time when a fit reaches
        the specified r^2 threshold value. This can be controlled via the
        coef_threshold argument.

        Solution:
        - Adjust the coef_threshold value. Note that it's highly
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
        )  # we can pass it higher params safely, they are ignored.
        disp = self.f.run(
            extend_by, coef_threshold, max_tries=max_tries, show_endpoint=show_endpoint,
        )
        disp = disp[3:]
        retval = [disp[i] * factorial(i + 1) for i in range(len(disp))]
        self._dispersion_array = retval
        return retval
