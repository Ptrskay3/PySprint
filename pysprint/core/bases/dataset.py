"""
This file implements the basic Dataset class.
"""
import json # for pretty printing dict
import warnings
from textwrap import dedent
from math import factorial

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysprint.core.bases.dataset_base import DatasetBase, C_LIGHT
from pysprint.core.preprocess import savgol, find_peak, convolution, cut_data, cwt
from pysprint.mpl_tools.normalize import DraggableEnvelope
from pysprint.utils.exceptions import *
from pysprint.mpl_tools.spp_editor import SPPEditor
from pysprint.utils import MetaData, run_from_ipython, find_nearest

__all__ = ['Dataset']


class Dataset(DatasetBase):
	"""
	Base class for the evaluating methods.
	"""
	meta = MetaData("""Additional info about the dataset""", copy=False)

	def __init__(self, x, y, ref=None, sam=None, meta=None):
		
		super().__init__()

		self.x = x
		self.y = y
		if ref is None:
			self.ref = []
		else:
			self.ref = ref 
		if sam is None:
			self.sam = []
		else:
			self.sam = sam
		self._is_normalized = False
		if not isinstance(self.x, np.ndarray):
			try:
				self.x = np.array(self.x)
				self.x.astype(float)
			except ValueError:
				raise DatasetError('Invalid type of data')
		if not isinstance(self.y, np.ndarray):
			try:
				self.y = np.array(self.y)
				self.y.astype(float)
			except ValueError:
				raise DatasetError('Invalid type of data')
		if not isinstance(self.ref, np.ndarray):
			try:
				self.ref = np.array(self.ref)
				self.ref.astype(float)
			except ValueError:
				pass # just ignore invalid arms
		if not isinstance(self.sam, np.ndarray):
			try:
				self.sam = np.array(self.sam)
				self.sam.astype(float)
			except ValueError:
				pass # just ignore invalid arms

		if len(self.ref) == 0:
			self.y_norm = self.y
			self._is_normalized = self._ensure_norm()

		else:
			self.y_norm = (self.y - self.ref - self.sam) / (2 * np.sqrt(self.sam * self.ref))
			self._is_normalized = True
		
		self.plotwidget = plt
		self.xmin = None
		self.xmax = None
		self.probably_wavelength = None
		self._check_domain()
		
		if meta is not None:
			self.meta = meta

		self._delay = None
		self._positions = None

		self._dispersion_array = None


	def phase_plot(self, exclude_GD=False):
		if not np.all(self._dispersion_array):
			raise ValueError('Dispersion must be calculated before plotting the phase.')

		coefs = np.array([self._dispersion_array[i]/factorial(i+1) for i in range(len(self._dispersion_array))])

		if exclude_GD:
			coefs[0] = 0

		phase_poly = np.poly1d(coefs[::-1], r=False)
		
		self.plotwidget.plot(self.x, phase_poly(self.x))
		self.plotwidget.grid()
		self.plotwidget.ylabel('$\Phi\, [rad]$')
		self.plotwidget.xlabel('$\omega \,[PHz]$')
		self.plotwidget.show()
		
	@property
	def delay(self):
		return self._delay

	@delay.setter
	def delay(self, value):
		self._delay = value

	@property
	def positions(self):
		return self._positions

	@positions.setter
	def positions(self, value):
		self._positions = value


	def _ensure_norm(self):
		idx = np.where((self.y_norm > 2))
		val = len(idx[0]) / len(self.y_norm)
		if val > 0.015: # this is a custom threshold, which often works..
			return False
		return True

	def scale_up(self):
		'''
		If the interferogram is normalized to [0, 1] interval, scale up to [-1, 1]
		with easy algerbra.. Just in case you need comparison, or any other purpose.
		'''
		self.y_norm = (self.y_norm - 0.5) * 2
		self.y = (self.y - 0.5) * 2

	def GD_lookup(self, reference_point=2.355, engine='cwt', silent=False, **kwargs):
		'''
		Quick GD lookup: it finds extremal points near the `reference_point` and returns
		an avarage value of 2*np.pi divided by distances between consecutive minimal or
		maximal values. Since it's relying on peak detection, the results may be irrelevant
		in some cases. If the parent class is `~pysprint.CosFitMethod`, then it will set the
		predicted value as inital parameter for fitting.
		'''
		if engine not in ('cwt', 'normal'):
			raise ValueError('Engine must be `cwt` or `normal`.')
		if engine == 'cwt':
			width = kwargs.pop('width', 35)
			floor_thres = kwargs.pop('floor_thres', 0.05)
			x_min, _, x_max, _ = self.detect_peak_cwt(width=width, floor_thres=floor_thres)

			#just validation
			_ = kwargs.pop('pmin', 0.1)
			_ = kwargs.pop('pmax', 0.1)
			_ = kwargs.pop('threshold', 0.35)

		else:
			pmin = kwargs.pop('pmin', 0.1)
			pmax = kwargs.pop('pmax', 0.1)
			threshold = kwargs.pop('threshold', 0.35)
			x_min, _, x_max, _ = self.detect_peak(pmin=pmin, pmax=pmax, threshold=threshold)

			# just validation
			_ = kwargs.pop('width', 10)
			_ = kwargs.pop('floor_thres', 0.05)

		if kwargs:
			raise TypeError(f'Invalid argument:{kwargs}')

		try:
			closest_val, idx1 = find_nearest(x_min, reference_point)
			m_closest_val, m_idx1 = find_nearest(x_max, reference_point)
		except ValueError:
			print('Prediction failed.\nSkipping.. ')
			return
		try:
			truncated = np.delete(x_min, idx1)
			second_closest_val, _ = find_nearest(truncated, reference_point)
		except IndexError:
			print('Prediction failed.\nSkipping.. ')
			return
		try:
			m_truncated = np.delete(x_max, m_idx1)
			m_second_closest_val, _ = find_nearest(m_truncated, reference_point)
		except IndexError:
			print('Prediction failed.\nSkipping.. ')
			return
		lowguess = 2*np.pi/np.abs(closest_val-second_closest_val)
		highguess = 2*np.pi/np.abs(m_closest_val-m_second_closest_val)
		if type(self).__name__ == 'CosFitMethod':
			self.params[3] = (lowguess+highguess)/2
		if not silent:
			print(f'The predicted GD is ± {((lowguess+highguess)/2):.5f} fs based on reference point of {reference_point}.')

	def _safe_cast(self):
		'''
		Return a copy of key attributes in order to prevent inplace modification.
		'''
		x, y, ref, sam = np.copy(self.x), np.copy(self.y), np.copy(self.ref), np.copy(self.sam)
		return x, y, ref, sam

	@staticmethod
	def wave2freq(value):
		'''
		Switches values between wavelength and angular frequency.
		'''
		return (2*np.pi*C_LIGHT)/value

	_dispatch = wave2freq.__func__

	@staticmethod
	def freq2wave(value):
		'''
		Switches values between angular frequency and wavelength.
		'''
		return Dataset._dispatch(value)

	def _check_domain(self):
		"""
		Checks the domain of data just by looking at x axis' minimal value.
		Units are obviously not added yet, we work in nm and PHz...
		"""
		if min(self.x) > 50:
			self.probably_wavelength = True
		else:
			self.probably_wavelength = False

	@classmethod
	def parse_raw(cls, basefile, ref=None, sam=None, skiprows=8,
		decimal=',', sep=';', meta_len=5):
		'''
		Dataset object alternative constructor. Helps to load in data just by giving the filenames
		in the target directory.

		Parameters:
		----------
		basefile: `str`
			base interferogram
			file generated by the spectrometer

		ref: `str`, optional
			reference arm's spectra
			file generated by the spectrometer

		sam: `str`, optional
			sample arm's spectra
			file generated by the spectrometer

		skiprows: `int`, optional
			Skip rows at the top of the file. Default is `8`.

		sep: `str`, optional
			The delimiter in the original interferogram file.
			Default is `;`.

		decimal: `str`, optional
			Character recognized as decimal separator in the original dataset. 
			Often `,` for European data.
			Default is `,`.

		meta_len: `int`, optional
			The first `n` lines in the original file containing the meta information
			about the dataset. It is parsed to be dict-like. If the parsing fails,
			a new entry will be created in the dictionary with key `unparsed`.
			Default is `5`.
		'''
		if skiprows < meta_len:
			warnings.warn(f'Skiprows is currently {skiprows}, but meta information is set to {meta_len} lines. This implies that either one is probably wrong.', PySprintWarning)
		with open(basefile) as file:
			comm = next(file).strip('\n').split('-')[-1].lstrip(' ')
			additional = (next(file).strip('\n').strip('\x00').split(':') for _ in range(1, meta_len))
			if meta_len != 0:
				cls.meta = {'comment': comm}
			try:
				for info in additional:
					cls.meta[info[0]] = info[1]
			except IndexError:
				cls.meta['unparsed'] = str(list(additional))
		df = pd.read_csv(basefile, skiprows=skiprows, sep=sep, decimal=decimal, usecols=[0,1], names=['x', 'y'])
		if (ref is not None and sam is not None):
			r = pd.read_csv(ref, skiprows=skiprows, sep=sep, decimal=decimal, usecols=[0,1], names=['x', 'y'])
			s = pd.read_csv(sam, skiprows=skiprows, sep=sep, decimal=decimal, usecols=[0,1], names=['x', 'y'])
			return cls(df['x'].values, df['y'].values, r['y'].values, s['y'].values)
		return cls(df['x'].values, df['y'].values)

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		if isinstance(self._delay, np.ndarray):
			pprint_delay = self._delay[0]
		elif isinstance(self._delay, float) or isinstance(self._delay, int):
			pprint_delay = self._delay
		else:
			pprint_delay = '-'
		string = dedent(f'''
		{type(self).__name__} object

		Parameters
		----------
		Datapoints: {len(self.x)}
		Normalized: {self._is_normalized}
		Predicted domain: {'wavelength' if self.probably_wavelength else 'frequency'}
		Delay value: {(str(pprint_delay) + ' fs') if np.all(self._delay) else 'Not given'}
		SPP position(s): {self._positions if np.all(self._positions) else 'Not given'}

		Metadata extracted from file
		----------------------------
		{json.dumps(self.meta, indent=4)}''')
		return string

	@property
	def data(self):
		"""
		Returns the *current* dataset as `pandas.DataFrame`.
		"""
		if self._is_normalized:
			try:
				self._data = pd.DataFrame({
				'x': self.x,
				'y': self.y,
				'sample': self.sam,
				'reference': self.ref,
				'y_normalized': self.y_norm
					})
			except ValueError:
				self._data = pd.DataFrame({
					'x': self.x,
					'y': self.y,
					})
		else:
			self._data = pd.DataFrame({
				'x': self.x,
				'y': self.y,
				})
		return self._data

	@property
	def is_normalized(self):
		"""
		Retuns whether the dataset is normalized.
		"""
		return self._is_normalized
	
	def chdomain(self):
		""" Changes from wavelength [nm] to ang. freq. [PHz] domain and vica versa."""
		self.x = (2*np.pi*C_LIGHT)/self.x
		self._check_domain()
		if type(self).__name__ == 'FFTMethod':
			self.original_x = self.x


	def detect_peak_cwt(self, width, floor_thres=0.05):
		x, y, ref, sam = self._safe_cast()
		xmax, ymax, xmin, ymin = cwt(x, y, ref, sam, width=width, floor_thres=floor_thres)
		return xmax, ymax, xmin, ymin

	def savgol_fil(self, window=5, order=3):
		"""
		Applies Savitzky-Golay filter on the dataset.

		Parameters:
		----------
		window: `int`
			Length of the convolutional window for the filter.
			Default is `10`.

		order: `int`
			Degree of polynomial to fit after the convolution.
			If not odd, it's incremented by 1. Must be lower than window.
			Usually it's a good idea to stay with a low degree, e.g 3 or 5.
			Default is 3.

		Notes:
		------
		If arms were given, it will merge them into the `self.y` and `self.y_norm` variables.
		Also applies a linear interpolation on dataset (and raises warning).
		"""
		self.x, self.y_norm = savgol(self.x, self.y, self.ref, self.sam, window=window, order=order)
		self.y = self.y_norm
		self.ref = []
		self.sam = []
		warnings.warn('Linear interpolation have been applied to data.', InterpolationWarning)
		
	
	def slice(self, start=None, stop=None):
		"""
		Cuts the dataset on x axis in this form: [start, stop]

		Parameters:
		----------
		start: `float`
			start value of cutting interval
			Not giving a value will keep the dataset's original minimum value.
			Note that giving `None` will leave original minimum untouched too.
			Default is `None`.

		stop: `float`
			stop value of cutting interval
			Not giving a value will keep the dataset's original maximum value.
			Note that giving `None` will leave original maximum untouched too.
			Default is `None`.

		Notes:
		------

		If arms were given, it will merge them into the `self.y` and `self.y_norm` variables.
		"""
		self.x, self.y_norm = cut_data(self.x, self.y, self.ref, self.sam, start=start, stop=stop)
		self.ref = []
		self.sam = []
		self.y = self.y_norm
		# Just to make sure it's correctly shaped. Later on we might delete this.
		if type(self).__name__ == 'FFTMethod':
			self.original_x = self.x
		self._is_normalized = self._ensure_norm()

	def convolution(self, window_length, std=20):
		"""
		Applies a convolution with a gaussian on the dataset

		Parameters:
		----------
		window_length: `int`
			Length of the gaussian window.

		std: `float`
			Standard deviation of the gaussian
			Default is `20`.

		Notes:
		------
		If arms were given, it will merge them into the `self.y` and `self.y_norm` variables.
		Also applies a linear interpolation on dataset (and raises warning).
		"""
		self.x, self.y_norm = convolution(self.x, self.y, self.ref, self.sam, window_length, standev=std)
		self.ref = []
		self.sam = []
		self.y = self.y_norm
		warnings.warn('Linear interpolation have been applied to data.', InterpolationWarning)


	def detect_peak(self, pmax=0.1, pmin=0.1, threshold=0.1, except_around=None):
		"""
		Basic algorithm to find extremal points in data using ``scipy.signal.find_peaks``.

		Parameters:
		----------

		pmax: `float`
			Prominence of maximum points.
			The lower it is, the more peaks will be found.
			Default is `0.1`.

		pmin: float
			Prominence of minimum points.
			The lower it is, the more peaks will be found.
			Default is `0.1`.

		threshold: `float`
			Sets the minimum distance (measured on y axis) required for a point to be
			accepted as extremal.
			Default is 0.

		except_around: interval (array or tuple), 
			Overwrites the threshold to be 0 at the given interval.
			format is `(lower, higher)` or `[lower, higher]`.
			Default is None.

		Returns:
		-------
		xmax: `array-like`
			x coordinates of the maximums

		ymax: `array-like`
			y coordinates of the maximums

		xmin: `array-like`
			x coordinates of the minimums

		ymin: `array-like`
			y coordinates of the minimums
		"""
		x, y, ref, sam = self._safe_cast()
		xmax, ymax, xmin, ymin = find_peak(x, y, ref, sam, proMax=pmax, proMin=pmin, threshold=threshold, except_around=except_around)
		return xmax, ymax, xmin, ymin
		
	def show(self):
		"""
		Draws a graph of the current dataset using matplotlib.
		"""
		if np.iscomplexobj(self.y):
			self.plotwidget.plot(self.x, np.abs(self.y))
		else:   
			try:
				self.plotwidget.plot(self.x, self.y_norm, 'r')
			except Exception:
				self.plotwidget.plot(self.x, self.y, 'r')
		self.plotwidget.grid()
		self.plotwidget.show()


	def normalize(self, filename=None, smoothing_level=0):
		'''
		Normalize the interferogram by finding upper and lower envelope
		on an interactive matplotlib editor.

		Parameters
		----------

		filename: `str`
			Save the normalized interferogram named by filename in the 
			working directory. If not given it will not be saved.
			Default None.

		Returns
		-------
		None
		'''
		if run_from_ipython():
			return '''It seems you run this code in IPython. Interactive plotting is not yet supported. Consider running it in the regular console.'''
		x, y, _, _ = self._safe_cast()
		if smoothing_level != 0:
			x, y = savgol(x, y, [], [], window=smoothing_level)
		_l_env = DraggableEnvelope(x, y, 'l')
		y_transform = _l_env.get_data()
		_u_env = DraggableEnvelope(x, y_transform, 'u')
		y_final = _u_env.get_data()
		self.y = y_final
		self.y_norm = y_final
		self._is_normalized = True
		self.plotwidget.title('Final')
		self.show()
		if filename:
			if not filename.endswith('.txt'):
				filename += '.txt'
			np.savetxt(filename, np.transpose([self.x, self.y]), delimiter=',')
			print(f'Successfully saved as {filename}')


	def open_SPP_panel(self):
		_spp = SPPEditor(self.x, self.y_norm)
		self.delay, self.positions = _spp.get_data()

	def emit(self):
		# validate if it's typed by hand..
		if not isinstance(self._positions, np.ndarray):
			self._positions = np.asarray(self.positions)
		if not isinstance(self.delay, np.ndarray):
			self.delay = np.ones_like(self.positions) * self.delay
		return self.delay, self.positions

	def set_SPP_data(self, delay, positions):
		if not isinstance(positions, np.ndarray):
			positions = np.asarray(positions)
		if not isinstance(delay, float):
			delay = float(delay)
		delay = np.ones_like(positions) * delay
		self._delay = delay
		self._positions = positions
