from math import factorial

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pysprint.utils import find_nearest
from pysprint.core.functions import _cosfit_config, cos_fit1


class FitOptimizer:
    """Class to help achieve better fitting results."""

    def __init__(self, x, y, ref, sam, reference_point, max_order=3):
        self.x = x
        self.y = y
        self.ref = ref
        self.sam = sam
        if not isinstance(self.x, np.ndarray):
            self.x = np.asarray(self.x)
        if not isinstance(self.y, np.ndarray):
            self.y = np.asarray(self.y)
        if not isinstance(self.ref, np.ndarray):
            try:
                self.ref = np.asarray(self.ref)
            except (ValueError, TypeError):
                pass  # we ignore because it might be optional if y is already normalized
        if not isinstance(self.sam, np.ndarray):
            try:
                self.sam = np.asarray(self.sam)
            except (ValueError, TypeError):
                pass  # we ignore because it might be optional if y is already normalized
        if len(self.ref) == 0:
            self._y_norm = self.y
        else:
            self._y_norm = (self.y - self.ref - self.sam) / (
                2 * np.sqrt(self.sam * self.ref)
            )
        self.reference_point = reference_point
        self.x -= self.reference_point
        self.func = cos_fit1
        self.p0 = [1, 1, 1, 1]
        self.popt = self.p0
        self._init_set = False
        self.counter = 0
        self.curr_order = 1
        self.max_order = max_order
        self.rest = None
        self.figure = plt.figure()

    def __del__(self):
        self.reference_point = 0
        self.x = None
        self.y = None
        self.ref = None
        self.sam = None

    def set_final_guess(self, GD, GDD=None, TOD=None, FOD=None, QOD=None):
        self.p0[3] = GD
        self.rest = []
        if GDD is not None:
            self.rest.append(GDD / 2)
        if TOD is not None:
            self.rest.append(TOD / 6)
        if FOD is not None:
            self.rest.append(FOD / 24)
        if QOD is not None:
            self.rest.append(QOD / 120)

    @property
    def user_guess(self):
        return self.rest

    # TODO: Make this update the plot, not redraw
    def update_plot(self):
        plt.clf()
        plt.plot(self.x, self._y_norm)
        plt.plot(self._x_curr, self._y_curr, "k", label="Affected data")
        plt.plot(
            self._x_curr,
            self.func(self._x_curr, *self.popt),
            "r--",
            label="Fit",
        )
        plt.grid()
        # plt.legend(loc='upper left')
        plt.draw()
        plt.xlabel("$\Delta\omega\, [PHz]$")
        plt.ylabel("I")
        plt.show()

    def set_initial_region(self, percent):
        """ Determines the initial region to fit"""
        self._init_set = True
        _, idx = find_nearest(self.x, 0)
        self._upper_bound = np.floor(idx + (percent / 2) * (len(self.x) + 1))
        self._lower_bound = np.floor(idx - (percent / 2) * (len(self.x) + 1))
        self._upper_bound = self._upper_bound.astype(int)
        self._lower_bound = self._lower_bound.astype(int)
        if self._lower_bound < 0:
            self._lower_bound = 0
        if self._upper_bound > len(self.x):
            self._upper_bound = len(self.x)
        self._x_curr = self.x[self._lower_bound : self._upper_bound]
        self._y_curr = self._y_norm[self._lower_bound : self._upper_bound]

    def _step_up_func(self):
        """
		Change the function to fit. Starts with first order
		and stepping up until max_order.
		"""
        if self.curr_order == self.max_order:
            return
        try:
            self.func = _cosfit_config[self.curr_order + 1]
        except KeyError as e:
            e.args = (e.args[0], "Order must be in [1, 5].")
            raise
        try:
            self.p0 = np.append(self.p0, self.rest[self.curr_order - 1])
        except (IndexError, ValueError):
            self.p0 = np.append(self.p0, 1)
        self.curr_order += 1

    def _extend_region(self, extend_by=0.1):
        """ Extends region of fit"""
        self._new_lower = np.floor(self._lower_bound - extend_by * len(self.x))
        self._new_upper = np.floor(self._upper_bound + extend_by * len(self.x))
        self._new_lower = self._new_lower.astype(int)
        self._new_upper = self._new_upper.astype(int)
        self._lower_bound = self._new_lower
        self._upper_bound = self._new_upper
        if self._new_lower < 0:
            self._new_lower = 0
        if self._new_upper > len(self.x):
            self._new_upper = len(self.x)
        self._x_curr = self.x[self._new_lower : self._new_upper]
        self._y_curr = self._y_norm[self._new_lower : self._new_upper]

    def _finetune(self):
        """
		Changes the last parameter randomly, we might get lucky..
		"""
        self.p0[-1] = float(np.random.uniform(-1, 1, 1)) * self.p0[-1]

    def _fit(self):
        try:
            if len(self._x_curr) == len(self.x):
                return True
            self.popt, self.pcov = curve_fit(
                self.func, self._x_curr, self._y_curr, p0=self.p0
            )
            self.p0 = self.popt
        except RuntimeError:
            self._finetune()
            self.popt, self.pcov = curve_fit(
                self.func, self._x_curr, self._y_curr, p0=self.p0
            )

    def _fit_goodness(self):
        """
		Coeffitient of determination a.k.a. r^2
		"""
        residuals = self._y_curr - self.func(self._x_curr, *self.popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self._y_curr - np.mean(self._y_curr)) ** 2)
        return 1 - (ss_res / ss_tot)

    def result_wrapper(self):
        labels = ("GD", "GDD", "TOD", "FOD", "QOD")
        params = self.p0[3:]
        for i, (label, param) in enumerate(zip(labels, params)):
            print(f"{label} = {(params[i]*factorial(i+1)):.5f} fs^{i + 1}")

    def run(
        self, r_extend_by, r_threshold, max_tries=5000, show_endpoint=True
    ):

        if not self._init_set:
            raise ValueError("Set the initial conditions.")
        print("This action might take a little time..")
        self._fit()
        while self._fit_goodness() > r_threshold:

            # self.figure.savefig(f'{self.counter}.eps')
            # self.update_plot()

            self._extend_region(r_extend_by)
            self._fit()
            self.counter += 1
            self._step_up_func()
            if self._fit() is True:
                if show_endpoint:
                    self.update_plot()
                    # self.figure.savefig(f'{self.counter}.eps')

                self.result_wrapper()
                print(f"with r^2 = {(self._fit_goodness()):.5f}.")
                return self.popt
            if self.counter == max_tries:
                if show_endpoint:
                    self.update_plot()
                print(
                    f"""Max tries ({max_tries}) reached.. try another initial params"""
                )
                return np.zeros_like(self.popt)

        while self._fit_goodness() < r_threshold:
            self._fit()
            # self._finetune()
            self.counter += 1
            if self.counter == max_tries:
                if show_endpoint:
                    self.update_plot()
                print(
                    f"""\nMax tries ({max_tries}) reached.. try another initial params"""
                )
                return np.zeros_like(self.popt)
