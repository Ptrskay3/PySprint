import numpy as np
import matplotlib.pyplot as plt

from pysprint.api.dataset import Dataset
from pysprint.core.optimizer import FitOptimizer
from pysprint.core.peak import EditPeak
from pysprint.utils import _maybe_increase_before_cwt, print_disp, run_from_ipython, findNearest as find_nearest
from pysprint.core.evaluate import cff_method


__all__ = ['CosFitMethod']


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


		# FIXME : DIFFERENT ERROR CASES SHOULD BE TAKEN CARE OF
	def predict(self, reference_point=2.355, pmax=0.5, pmin=0.5, threshold=0.35):
		x_min, _, x_max, _ = self.detect_peak(pmax=pmax, pmin=pmin, threshold=threshold)
		try:
			closest_val, idx1 = find_nearest(x_min, reference_point)
			m_closest_val, m_idx1 = find_nearest(x_max, reference_point)
		except ValueError:
			print('Prediction failed.\nSkipping.. ')
			return
		truncated = np.delete(x_min, idx1)
		second_closest_val, _ = find_nearest(truncated, reference_point)
		m_truncated = np.delete(x_max, m_idx1)
		m_second_closest_val, _ = find_nearest(m_truncated, reference_point)
		lowguess = 2*np.pi/np.abs(closest_val-second_closest_val)
		highguess = 2*np.pi/np.abs(m_closest_val-m_second_closest_val)
		self.params[3] = (lowguess+highguess)/2
		print(f'The predicted GD is ± {((lowguess+highguess)/2):.5f} based on reference point of {reference_point}.')

	def set_max_tries(self, value):
		"""
		Overwrite the default scipy maximum try setting to fit the curve.
		"""
		self.mt = value

	def adjust_offset(self, value):
		"""
		Initial guess for offset
		"""
		self.params[0] = value

	def adjust_amplitude(self, value):
		"""
		Initial guess for amplitude
		"""
		self.params[1] = value

	def guess_GD(self, value):
		"""
		Initial guess for GD in fs
		"""
		self.params[3] = value

	def guess_GDD(self, value):
		"""
		Initial guess for GDD in fs^2
		"""
		self.params[4] = value / 2

	def guess_TOD(self, value):
		"""
		Initial guess for TOD in fs^3
		"""
		self.params[5] = value / 6

	def guess_FOD(self, value):
		"""
		Initial guess for FOD in fs^4
		"""
		self.params[6] = value / 24

	def guess_QOD(self, value):
		"""
		Initial guess for QOD in fs^5
		"""
		self.params[7] = value / 120

	def set_max_order(self, order):
		"""
		Sets the maximum order of dispersion to look for.
		Should be called after guessing the initial parameters.

		Parameters:
		----------

		order: int
			maximum order of dispersion to look for. Must be in [1, 5]
		"""
		if order > 5 or order < 1:
			print(f'Order should be an in integer from [1,5], currently {order} is given')
		try:
			int(order)
		except ValueError:
			print(f'Order should be an in integer from [1,5], currently {order} is given')
		order = 6 - order
		for i in range(1, order):
			self.params[-i] = 0

	@print_disp
	def calculate(self, reference_point):
		""" 
		Cosine fit's calculate function.

		Parameters:
		----------
		reference_point: float
			reference point on x axis

		Returns:
		-------

		dispersion: array-like
			[GD, GDD, TOD, FOD, QOD]

		dispersion_std: array-like
			standard deviations due to uncertanity of the fit
			[GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

		fit_report: lmfit report
			WILL BE IMPLEMENTED


		Notes:
		------

		Decorated with print_disp, so the results are immediately printed without explicitly saying so.
		"""
		dispersion, self.fit = cff_method(self.x, self.y, self.ref, self.sam, 
			ref_point=reference_point, p0=self.params, maxtries=self.mt)
		dispersion = list(dispersion)
		while len(dispersion) < 5:
			dispersion.append(0)
		return dispersion, [0, 0, 0, 0, 0], 'Fit report for CFF not supported yet.'

	def plot_result(self):
		"""
		If the fitting happened, draws the fitted curve on the original dataset.
		Also prints the coeffitient of determination of the fit (a.k.a. r^2).
		"""
		try:
			residuals = self.y_norm - self.fit
			ss_res = np.sum(residuals**2)
			ss_tot = np.sum((self.y_norm - np.mean(self.y_norm))**2)
			print('r^2 = ' + str(1 - (ss_res / ss_tot)))
		except Exception:
			pass
		if self.fit is not None:
			self.plotwidget.plot(self.x, self.fit, 'k--', label='fit', zorder=99)
			self.plotwidget.legend()
			self.show()
		else:
			self.show()

	def optimizer(self, reference_point, order=3, initial_region_ratio=0.1,
		extend_by=0.1, coef_threshold=0.3, max_tries=5000, show_endpoint=True):
		"""
		Cosine fit optimizer. It's based on adding new terms to fit function successively
		until we reach the max_order.

		Notes
		-----
		If the fit fails some parameters must be tweaked in order to achieve results.
		There is a list below with issues, its suspected reasons and solutions.

		**SciPy raises OptimizeWarning and the affected area is small or not showing
		  any fit

		Reasons:
		- Completely wrong initial GD guess (or lack of guessing).
		- Too broad inital region, so that the optimizer cannot find a suitable fit.
		  This usually happens when the used data is large, or the spectral resolution
		  is high.

		Solution:
		- Provide better inital guess for GD.
		- Lower the inital_region_ratio.

		**SciPy raises OptimizeWarning and the affected area is bigger

		Reasons:
		- When the optimizer steps up with order it also extends the region of fit.
		This error usually present when the region of fit is too quickly growing.

		Solution:
		- Lower extend_by argument.

		**The optimizer is finished, but wrong fit is produced.

		Reasons:
		- We measure the goodness of fit with r^2 value. To allow this
		optimizer to smoothly find appropriate fits even for noisy datasets
		it's a good practice to keep the r^2 a lower value, such as the default 0.3.
		The way it works is we step up in order of fit (until max order) and extend
		region every time when a fit reaches the specified r^2 threshold value.
		This can be controlled via the coef_threshold argument.

		Solution:
		- Adjust the coef_threshold value. Note that it's highly recommended not to
		set a higher value than 0.6.
		"""

		x, y, ref, sam = self._safe_cast()
		self.f = FitOptimizer(x, y, ref, sam, reference_point=reference_point,
		max_order=order)
		self.f.set_initial_region(initial_region_ratio)
		self.f.set_final_guess(GD=self.params[3], GDD=self.params[4], TOD=self.params[5],
		FOD=self.params[6], QOD=self.params[7]) # we can pass it higher params safely, they are ignored.
		self.f.run_loop(extend_by, coef_threshold, max_tries=max_tries, show_endpoint=show_endpoint)