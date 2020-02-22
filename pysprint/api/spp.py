import numpy as np
import matplotlib.pyplot as plt

from pysprint.api.dataset import Dataset
from pysprint.utils import print_disp
from pysprint.core.evaluate import spp_method


__all__ = ['SPPMethod']

class SPPMethod(Dataset):
	"""
	Basic interface for Stationary Phase Point Method. It will be improved later.
	"""
	raw = False

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.om = None
		self.de = None
		self.bf = None

	@classmethod
	def from_raw(cls, omegas, delays):
		"""
		Alternative constructor to work with matching pairs of data.

		Parameters:
		----------

		omegas: array-like
			the angular frequency values at SPP positions

		delays: array-like
			the delay of the corresponding omega 

		"""
		cls.raw = True
		return cls(omegas, delays)

	@print_disp
	def calculate(self, reference_point, order):
		""" 
		SPP's calculate function.

		Parameters:
		----------

		reference_point: float
			reference point on x axis

		fit_order: int
			Polynomial (and maximum dispersion) order to fit. Must be in [1,4].

		Returns:
		-------

		dispersion: array-like
			[GD, GDD, TOD, FOD, QOD]

		dispersion_std: array-like
			standard deviations due to uncertanity of the fit
			[GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

		bf: array-like
			if lmfit is available, the best fitting curve

		Notes:
		------

		Decorated with print_disp, so the results are immediately printed without explicitly saying so.
		"""
		if self.raw:
			_, _, dispersion, dispersion_std, self.bf = spp_method(
				self.y, self.x, reference_point=reference_point, fitOrder=order, from_raw=True
				)
			self.om = self.x
			self.de = self.y
		else:
			self.om, self.de, dispersion, dispersion_std, self.bf = spp_method(
				self.y, self.x, fitOrder=order, from_raw=False
				)
		dispersion = list(dispersion)
		dispersion_std = list(dispersion_std)
		while len(dispersion)<5:
			dispersion.append(0)
		while len(dispersion_std)<5:
			dispersion_std.append(0)
		return dispersion, dispersion_std, self.bf

	def plot_result(self):
		"""
		If the fitting happened, draws the fitted curve on the original dataset.
		"""
		self.plotwidget.plot(self.om, self.de, 'o')
		try:
			self.plotwidget.plot(self.om, self.bf, 'r--', zorder=1)
		except Exception:
			pass
		self.plotwidget.show()
