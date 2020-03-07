import numpy as np
from scipy.fftpack import fftshift

from pysprint.api.dataset import Dataset
from pysprint.utils import print_disp
from pysprint.core.evaluate import (fft_method,	cut_gaussian, ifft_method, 
	 args_comp, gaussian_window)
from pysprint.api.exceptions import *

__all__ = ['FFTMethod']

class FFTMethod(Dataset):
	"""
	Basic interface for the Fourier transform method.
	# FIXME: bug when calling show_graph on calulcate method: it opens up a clean figure.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		#making sure it's not normalized
		if self._is_normalized:
			self.y_norm = self.y
			self._is_normalized = False
		self.original_x = self.x
		self.at = None
		self.std = None
		self.window_order = None
		self._ifft_called_first = False

	def shift(self, axis='x'):
		"""
		Equivalent to scipy.fftpack.fftshift, but it's easier to
		use this function instead, because we don't need to explicitly
		call the class' x and y attribute.
		
		Parameter(s):
		------------
		axis: str, default is 'x'
			either 'x', 'y' or 'both'
		"""
		if axis == 'x':
			self.x = fftshift(self.x)
		elif axis == 'y':
			self.y = fftshift(self.y)
		elif axis == 'both':
			self.y = fftshift(self.y)
			self.x = fftshift(self.x)
		else:
			raise ValueError(f'axis should be either x, y or both, currently {axis} is given.')

	def ifft(self, interpolate=True):
		"""
		Applies ifft to the dataset.
		
		Parameter(s):
		------------

		interpolate: bool, default is True
			Whether to apply linear interpolation on the dataset before transforming.
		"""
		self._ifft_called_first = True
		self.x, self.y = ifft_method(self.x, self.y, interpolate=interpolate)

	def fft(self):
		"""
		Applies fft to the dataset.
		If ifft was not called first, inaccurate results might happen. It will be fixed later on.
		Check calculate function's docstring for more detail.
		"""
		if not self._ifft_called_first:
			warnings.warn('This module is designed to call ifft before fft, so inconsistencies might occur when calling fft first. Consider using numpys fft package with your own logic. This functionality will be added later on.', FourierWarning)
		self.x, self.y = fft_method(self.original_x, self.y)

	def window(self, at, fwhm, window_order=6, plot=True):
		"""
		Draws a gaussian window on the plot with the desired parameters.
		The maximum value is adjusted for the dataset mostly for visibility reasons.
		You should explicitly call self.show() after this function is set.

		Parameters:
		----------

		at: float
			maximum of the gaussian curve

		fwhm: float
			Full width at half maximum of the gaussian

		window_order: int, default is 6
			Order of the gaussian curve.
			If not even, it's incremented by 1 for safety reasons.
		"""
		self.at = at
		self.fwhm = fwhm
		self.window_order = window_order
		gaussian = gaussian_window(self.x, self.at, self.fwhm, self.window_order)
		self.plotwidget.plot(self.x, gaussian*max(abs(self.y)), 'r--')
		if plot:
			self.show()

	def apply_window(self):
		"""
		If window function is correctly set, applies changes to the dataset.
		"""
		self.plotwidget.clf()
		self.plotwidget.cla()
		self.plotwidget.close()
		self.y = cut_gaussian(self.x, self.y, spike=self.at, fwhm=self.fwhm, win_order=self.window_order)
		
	@print_disp
	def calculate(self, reference_point, order, show_graph=False):
		""" 
		FFTMethod's calculate function.

		Parameters:
		----------

		reference_point: float
			reference point on x axis

		fit_order: int
			Polynomial (and maximum dispersion) order to fit. Must be in [1,5].

		show_graph: bool, optional
			shows a the final graph of the spectral phase and fitted curve.

		Returns:
		-------

		dispersion: array-like
			[GD, GDD, TOD, FOD, QOD]

		dispersion_std: array-like
			standard deviations due to uncertanity of the fit
			[GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

		fit_report: lmfit report
			if lmfit is available, the fit report

		Notes:
		------

		Decorated with print_disp, so the results are immediately printed without explicitly saying so.

		Currently the x-axis transformation is sloppy, because we cache the original x axis and not transforming it
		backwards. In addition we need to keep track of interpolation and zero-padding too.
		Currently the transforms are correct only if first ifft was used.
		For now it's doing okay: giving good results. 
		For consistency we should still implement that a better way later.
		"""
		dispersion, dispersion_std, fit_report = args_comp(
			self.x, self.y, ref_point=reference_point, fit_order=order, show_graph=show_graph
			)
		self._dispersion_array = dispersion
		return dispersion, dispersion_std, fit_report
