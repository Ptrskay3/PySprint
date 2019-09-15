"""
NOT PART OF THE MAIN PROGRAM YET.

"""

import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


def find_nearest(array, value):
	#Finds the nearest element to the given value in the array
	#returns tuple: (element, element's index)
	
    array = np.asarray(array)
    idx = (np.abs(value - array)).argmin()
    return array[idx], idx


def cos_fit(x,c0, c1, b0, b1, b2, b3):
	"""
	Auxiliary function for Phase Modulated Cosine Function Fit 
	b1 = GD
	b2 = GDD / 2
	b3 = TOD / 6
	b4 = FOD / 24
	b5 = QOD / 120
	"""
	return c0 + c1*np.cos(b0 + b1*x + b2*x**2 + b3*x**3)

class FitOptimizer(object):

	def __init__(self, x, y, ref, sam, func = None, p0 = []):
		self.x = x
		self.y = y
		self.ref = ref
		self.sam = sam
		if not isinstance(self.x, np.ndarray):
			try:
				self.x = np.asarray(self.x)
			except:
				raise
		if not isinstance(self.y, np.ndarray):
			try:
				self.y = np.asarray(self.y)
			except:
				raise
		if not isinstance(self.ref, np.ndarray):
			try:
				self.ref = np.asarray(self.ref)
			except:
				raise
		if not isinstance(self.sam, np.ndarray):
			try:
				self.sam = np.asarray(self.sam)
			except:
				raise
		self._y_norm = (self.y - self.ref - self.sam)/(2*np.sqrt(self.sam*self.ref))
		self.func = func
		self.p0 = [1, 1, 1, 1, 1, 1]
		self.popt = p0
		self._init_set = False
		

	def set_initial_region(self, percent, center):
		""" Determines the initial region to fit"""
		self._init_set = True
		_, idx = find_nearest(self.x, center)
		self.upper_bound = np.floor(idx + (percent/2)*(len(self.x) + 1))
		self.lower_bound = np.floor(idx - (percent/2)*(len(self.x) + 1))
		self.upper_bound = self.upper_bound.astype(int)
		self.lower_bound = self.lower_bound.astype(int)
		if self.lower_bound < 0:
			self.lower_bound = 0
		if self.upper_bound > len(self.x):
			self.upper_bound = len(self.x)

		self._x_curr = self.x[self.lower_bound:self.upper_bound]
		self._y_curr = self._y_norm[self.lower_bound:self.upper_bound]
		

	def _extend_region(self, extend_by = 0.2):
		self.new_lower = np.floor(self.lower_bound - extend_by*len(self.x))
		self.new_upper = np.floor(self.upper_bound + extend_by*len(self.x))
		self.new_lower = self.new_lower.astype(int)
		self.new_upper = self.new_upper.astype(int)
		self.lower_bound = self.new_lower
		self.upper_bound = self.new_upper
		if self.new_lower < 0:
			self.new_lower = 0
		if self.new_upper > len(self.x):
			self.new_upper = len(self.x)
		self._x_curr = self.x[self.new_lower:self.new_upper]
		self._y_curr = self._y_norm[self.new_lower:self.new_upper]

	def _make_fit(self):
		""" Makes fit  """
		try:
			if len(self._x_curr) == len(self.x):
				return True
			self.popt, self.pcov = curve_fit(self.func, self._x_curr, self._y_curr, maxfev = 200000, p0 = self.p0)
			self.p0 = self.popt 
		except RuntimeError:
			self.p0[:3] = self.popt[:3] + np.random.normal(0, 100, self.popt.shape-3)
			self.popt, self.pcov = curve_fit(self.func, self._x_curr, self._y_curr, maxfev = 200000, p0 = self.p0)


	# def _perturb_param(self, which = 2):
		# self.p0 = self.popt
		# if which == 0:
			# pass
		# else:
			# self.p0[which] = np.random.normal(0, 10**(which+1), 1)



	def _fit_goodness(self):
		residuals = self._y_curr - self.func(self._x_curr, *self.popt)
		ss_res = np.sum(residuals**2)
		ss_tot = np.sum((self._y_curr - np.mean(self._y_curr))**2)
		return 1 - (ss_res / ss_tot)

	
	def show_fit(self, time = 0.2):
		try:
			plt.figure(figsize = (8,6))
			plt.plot(self.x, self._y_norm,'k',label = 'Original')
			plt.plot(self._x_curr, self._y_curr, 'bo', label = 'Affected data')
			plt.plot(self._x_curr, self.func(self._x_curr, *self.popt), 'r--', label = 'Fit')
			plt.grid()
			plt.legend()
			plt.show(block=False)
			plt.pause(time)
			plt.close()
		except:
			print('aborting..')
			# plt.figure()
			# plt.plot(self._x_curr, self._y_curr)
			# plt.show()
			# print('passed..')

	def run_loop(self, r_extend_by, r_threshold, max_tries = 1000, show_steps = False):
		if self._init_set == False:
			raise ValueError('Set the initial conditions.')
		self._make_fit()
		# self._perturb_param(0)
		# self.show_fit()
		counter = 1
		# good_step = 0
		while self._fit_goodness() > r_threshold:
			# good_step += 1
			self._extend_region(r_extend_by)
			self._make_fit()
			if show_steps:
				self.show_fit()
			counter +=1
			if counter % 1000 == 0:
				print('Currect tries:', counter)
			if self._make_fit() == True:
				self.show_fit(50)
				print('The overall fit goodness is: ', self._fit_goodness())
				print('The params were:', self.popt)
				print('steps :', counter)
				break
			if counter == max_tries:
				self.show_fit(50)
				print('Max tries ({}) reached.. try another initial params.'.format(max_tries))
				break
				
		while self._fit_goodness() < r_threshold:
			self._make_fit()
			# self._perturb_param(good_step+1)
			counter +=1
			if counter % 1000 == 0:
				print('Currect tries:', counter)
			# if counter % 50 == 0:
				# self.show_fit()
			if counter == max_tries:
				self.show_fit(50)
				print('Max tries ({}) reached.. try another initial params.'.format(max_tries))
				break


a, b, c, d = np.loadtxt('tesztt.txt', delimiter = ',', unpack = True ) # f.set_initial_region(0.15, 2.5) #f.run_loop(r_extend_by = 0.05, r_threshold = 0.79, max_tries = 20000, show_steps = True)
# a, b, c, d = np.loadtxt('FOD.txt', delimiter = ',', unpack = True )

f = FitOptimizer(a,b,c,d, func = cos_fit)
f.set_initial_region(0.15, 2.50)
f.run_loop(r_extend_by = 0.1, r_threshold = 0.85, max_tries = 10000, show_steps = True)














