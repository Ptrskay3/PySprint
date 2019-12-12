"""
NEED TO IMPLEMENT:
 - better fit method (it's actually performing good with real datasets)
 - currently compatibility with GUI is dropped.. we should fix that.
 - clean up the code, it's messy
"""
from math import factorial

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from pysprint.utils import findNearest as find_nearest
from pysprint.core.evaluate import cos_fit1, cos_fit2, cos_fit3, cos_fit4, cos_fit5

_func_config = {i : eval(f'cos_fit{i}') for i in range(1, 6)}

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
				pass # we ignore because it might be optional if y is already normalized
		if not isinstance(self.sam, np.ndarray):
			try:
				self.sam = np.asarray(self.sam)
			except (ValueError, TypeError):
				pass # we ignore because it might be optional if y is already normalized
		if len(self.ref) == 0:
			self._y_norm = self.y
		else:
			self._y_norm = (self.y - self.ref - self.sam)/(2*np.sqrt(self.sam*self.ref))
		self.x -= reference_point
		self.reference_point = reference_point
		self.func = cos_fit1
		self.p0 = [1, 1, 1, 1]
		self.popt = self.p0
		self._init_set = False
		self.counter = 0
		self.curr_order = 1
		self.max_order = max_order
		self.rest = None
		self.figure = plt.figure

	def set_final_guess(self, GD, GDD=None, TOD=None, FOD=None, QOD=None):
		self.p0[3] = GD
		self.rest = []
		if GDD is not None:
			self.rest.append(GDD/2)
		if TOD is not None:
			self.rest.append(TOD/6)
		if FOD is not None:
			self.rest.append(FOD/24)
		if QOD is not None:
			self.rest.append(QOD/120)

	@property
	def user_guess(self):
		return self.rest

	def update_plot(self):
		plt.clf()
		plt.plot(self.x, self._y_norm)
		plt.plot(self._x_curr, self._y_curr, 'k', label='affected')
		plt.plot(self._x_curr, self.func(self._x_curr, *self.popt), 'r--', label='fit')
		plt.grid()
		plt.legend(loc='upper left')
		plt.draw()
		plt.show()

	def set_initial_region(self, percent):
		""" Determines the initial region to fit"""
		self._init_set = True
		_, idx = find_nearest(self.x, 0)
		self._upper_bound = np.floor(idx + (percent/2)*(len(self.x) + 1))
		self._lower_bound = np.floor(idx - (percent/2)*(len(self.x) + 1))
		self._upper_bound = self._upper_bound.astype(int)
		self._lower_bound = self._lower_bound.astype(int)
		if self._lower_bound < 0:
			self._lower_bound = 0
		if self._upper_bound > len(self.x):
			self._upper_bound = len(self.x)
		self._x_curr = self.x[self._lower_bound:self._upper_bound]
		self._y_curr = self._y_norm[self._lower_bound:self._upper_bound]

	def _step_up_func(self):
		"""
		Change the function to fit. Starts with first order
		and stepping up until max_order.
		"""
		if self.curr_order == self.max_order:
			return
		self.func = _func_config[self.curr_order + 1]
		try:
			self.p0 = np.append(self.p0, self.rest[self.curr_order-1])
		except (IndexError,ValueError):
			self.p0 = np.append(self.p0, 1)
		self.curr_order += 1

	def _extend_region(self, extend_by=0.1):
		""" Extends region of fit"""
		self._new_lower = np.floor(self._lower_bound - extend_by*len(self.x))
		self._new_upper = np.floor(self._upper_bound + extend_by*len(self.x))
		self._new_lower = self._new_lower.astype(int)
		self._new_upper = self._new_upper.astype(int)
		self._lower_bound = self._new_lower
		self._upper_bound = self._new_upper
		if self._new_lower < 0:
			self._new_lower = 0
		if self._new_upper > len(self.x):
			self._new_upper = len(self.x)
		self._x_curr = self.x[self._new_lower:self._new_upper]
		self._y_curr = self._y_norm[self._new_lower:self._new_upper]

	def _finetune(self):
		"""
		Changes the last parameter randomly, we might get lucky..
		"""
		self.p0[-1] = np.random.normal(0, 1, 1) * self.p0[-1]

	def _make_fit(self):
		try:
			if len(self._x_curr) == len(self.x):
				return True
			self.popt, self.pcov = curve_fit(
				self.func, self._x_curr, self._y_curr, maxfev=10000, p0=self.p0
				)
			self.p0 = self.popt
		except RuntimeError:
			self._finetune()
			self.popt, self.pcov = curve_fit(
				self.func, self._x_curr, self._y_curr, maxfev=10000, p0=self.p0
				)


	def _fit_goodness(self):
		"""
		Coeffitient of determination a.k.a. r^2
		"""
		residuals = self._y_curr - self.func(self._x_curr, *self.popt)
		ss_res = np.sum(residuals**2)
		ss_tot = np.sum((self._y_curr - np.mean(self._y_curr))**2)
		return 1 - (ss_res / ss_tot)

	def result_wrapper(self):
		labels = ('GD', 'GDD', 'TOD', 'FOD', 'QOD')
		params = self.p0[3:]
		for i, (label, param) in enumerate(zip(labels, params)):
			print(f'{label} = {params[i]*factorial(i+1)} fs^{i + 1}')

	def run_loop(
		self, r_extend_by, r_threshold,
		max_tries=5000, show_endpoint=True
		):
		print('__Optimizer__')
		pbar = tqdm(total=max_tries)
		if self._init_set is False:
			raise ValueError('Set the initial conditions.')
		self._make_fit()
		while self._fit_goodness() > r_threshold:
			self._extend_region(r_extend_by)
			self._make_fit()
			self.counter += 1
			self._step_up_func()
			pbar.update(1)
			if self._make_fit() is True:
				pbar.close()
				if show_endpoint:
					self.update_plot()
				self.result_wrapper()
				print(f'with r^2 = {self._fit_goodness()}.')
				return self.popt
			if self.counter == max_tries:
				if show_endpoint:
					self.update_plot()
				pbar.close()
				print(f'''Max tries ({max_tries}) reached.. try another initial params''')
				return np.zeros_like(self.popt)

		while self._fit_goodness() < r_threshold:
			self._make_fit()
			self.counter += 1
			pbar.update(1)
			if self.counter == max_tries:
				if show_endpoint:
					self.update_plot()
				pbar.close()
				print(f'''\nMax tries ({max_tries}) reached.. try another initial params''')
				return np.zeros_like(self.popt)
