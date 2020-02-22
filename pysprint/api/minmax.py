import numpy as np
from pysprint.api.dataset import Dataset
from pysprint.core.peak import EditPeak
from pysprint.utils import _maybe_increase_before_cwt, print_disp, run_from_ipython
from pysprint.core.evaluate import min_max_method

__all__ = ['MinMaxMethod']

class MinMaxMethod(Dataset):
	"""
	Basic interface for Minimum-Maximum Method.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	# TODO: fix docstring
	def init_edit_session(
		self, pmax=0.1, pmin=0.1, threshold=0, 
		except_around=None, engine='normal', width=10
		):
		""" Function to initialize peak editing on a plot.
		Right clicks will delete the closest point, left clicks 
		will add a new point. Just close the window when finished.
		
		Parameters:
		----------

		pmax: float, default is 0.1
			prominence of maximum points
			the lower it is, the more peaks will be found

		pmin: float, default is 0.1
			prominence of minimum points
			the lower it is, the more peaks will be found

		threshold: float, default is 0
			sets the minimum distance (measured on y axis) required for a point to be
			accepted as extremal

		except_around: interval (array or tuple), default is None
			Overwrites the threshold to be 0 at the given interval.
			format is (lower, higher) or [lower, higher].

		Notes:
		------

		Currently this function is disabled when running it from IPython.
		"""
		if run_from_ipython():
			return '''It seems you run this code in IPython. Interactive plotting is not yet supported. Consider running it in the regular console.'''
		engines = ('cwt', 'normal', 'slope')
		if engine not in engines:
			raise ValueError(f'Engine must be in {str(engines)}')
		if engine == 'normal':
			_x, _y, _xx, _yy = self.detect_peak(
			pmax=pmax, pmin=pmin, threshold=threshold, except_around=except_around
			)

		elif engine == 'slope':
			x, _, _, _ = self._safe_cast()
			y = np.copy(self.y_norm)
			if _maybe_increase_before_cwt(y):
				y += 2
			_, lp, lloc = calc_envelope(y, np.arange(len(y)), 'l')
			_, up, uloc = calc_envelope(y, np.arange(len(y)), 'u')
			lp -= 2
			up -= 2
			_x, _xx = x[lloc], x[uloc]
			_y, _yy = lp, up

		elif engine == 'cwt':
			_x, _y, _xx, _yy = self.detect_peak_cwt(width)
		_xm = np.append(_x, _xx)
		_ym = np.append(_y, _yy)

		try:
			_editpeak = EditPeak(self.x, self.y_norm, _xm, _ym)
		except ValueError:
			_editpeak = EditPeak(self.x, self.y, _xm, _ym)
		# automatically propagate these points to the mins and maxes
		# just in case the default argrelextrema is definitely not called in evaluate.py/min_max_method:
		self.xmin = _editpeak.get_dat[0][:len(_editpeak.get_dat[0])//2]
		self.xmax = _editpeak.get_dat[0][len(_editpeak.get_dat[0])//2:] 
		print(f'In total {len(_editpeak.get_dat[0])} extremal points were recorded.')
		return _editpeak.get_dat[0]

	@print_disp
	def calculate(self, reference_point, order, show_graph=False):
		""" 
		MinMaxMethod's calculate function.

		Parameters:
		----------

		reference_point: float
			reference point on x axis

		fit_order: int
			Polynomial (and maximum dispersion) order to fit. Must be in [1,5].

		show_graph: bool
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
		"""
		dispersion, dispersion_std, fit_report = min_max_method(
			self.x, self.y, self.ref, self.sam, ref_point=reference_point,
			maxx=self.xmax, minx=self.xmin, fitOrder=order, showGraph=show_graph
			)
		return dispersion, dispersion_std, fit_report