"""
This file is the main API to use Interferometry without the PyQt5 UI.
"""
import sys
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.fftpack import fftshift

from pysprint.core.evaluate import min_max_method, cff_method, fft_method, cut_gaussian, ifft_method, spp_method, args_comp, gaussian_window
from pysprint.core.dataedits import savgol, find_peak, convolution, cut_data
from pysprint.core.generator import generatorFreq, generatorWave
from pysprint.core.cff_fitting import FitOptimizer
from pysprint.core.peak import EditPeak
from pysprint.utils import print_disp, run_from_ipython, findNearest as find_nearest


__all__ = ['Generator', 'Dataset', 'MinMaxMethod', 'CosFitMethod', 'SPPMethod', 'FFTMethod']

C_LIGHT = 299.793 # nm/fs

# setting up the IPython notebook
if run_from_ipython():
	plt.rcParams['figure.figsize'] = [15, 5]

class DatasetError(Exception):
	"""
	This error is raised when invalid type of data encountered when initializing 
	a dataset or inherited object.
	"""
	pass


class InterpolationWarning(Warning):
	"""
	This warning is raised when a function applies linear interpolation on the data.
	"""
	pass


class FourierWarning(Warning):
	"""
	This warning is raised when FFT is called first instead of IFFT.
	Later on it will be improved. 
	For more details see help(pysprint.FFTMethod.calculate)
	"""
	pass


class BaseApp(object):
	def __init__(self):
		pass

	def run(self):
		"""
		Opens up the GUI with the loaded data.
		"""
		from pysprint.logic import MainProgram
		try:
			from PyQt5 import QtWidgets
		except ImportError:
			print('PyQt5 is essential for the UI. Use the API instead.')
		print('Building up UI..')
		app = QtWidgets.QApplication(sys.argv)
		main_app = MainProgram()
		main_app.showMaximized()
		main_app.a = self.x
		main_app.b = self.y
		main_app.samY = self.sam
		main_app.refY = self.ref
		if main_app.settings.value('show') == 'True':
			main_app.msgbox.exec_()
			if main_app.cb.isChecked():
				main_app.settings.setValue('show', False)
		else:
			pass
		main_app.redraw_graph()
		main_app.fill_table()
		main_app.track_stats()
		sys.exit(app.exec_())
		

class Generator(BaseApp):
	"""
	Basic dataset generator.
	"""
	def __init__(self, start, stop, center, delay=0,
		GD=0, GDD=0, TOD=0, FOD=0, QOD=0, resolution=0.1,
	 	delimiter=',', pulse_width=10, normalize=False, chirp=0):
		self.start = start
		self.stop = stop
		self.center = center
		self.delay = delay
		self.GD = GD
		self.GDD = GDD
		self.TOD = TOD
		self.FOD = FOD
		self.QOD = QOD
		self.resolution = resolution
		self.delimiter = delimiter
		self.pulse_width = pulse_width
		self.chirp = chirp
		self.normalize = normalize
		self.x = np.array([])
		self.y = np.array([])
		self.ref = np.array([])
		self.sam = np.array([])
		self.plotwidget = plt
		self.is_wave = False
		
	def __str__(self):
		return f'''Generator({self.start}, {self.stop}, {self.center}, delay = {self.delay},
				   GD={self.GD}, GDD={self.GDD}, TOD={self.TOD}, FOD={self.FOD}, QOD={self.QOD}, resolution={self.resolution}, 
				   delimiter={self.delimiter}, pulse_width={self.pulseWidth}, normalize={self.normalize})'''

	def _check_norm(self):
		"""
		Does the normalization when we can.
		"""
		if len(self.ref) != 0:
			self._y =  (self.y - self.ref - self.sam)/(2*np.sqrt(self.sam*self.ref))

	def generate_freq(self):
		"""
		Use this to generate the spectrogram in ang. frequency domain.
		"""
		self.x, self.y, self.ref, self.sam = generatorFreq(self.start, self.stop, self.center, self.delay, self.GD,
			self.GDD, self.TOD, self.FOD, self.QOD,
			self.resolution, self.delimiter, self.pulse_width, self.normalize, self.chirp)

	def generate_wave(self):
		"""
		Use this to generate the spectrogram in wavelength domain.
		"""
		self.is_wave = True
		self.x, self.y, self.ref, self.sam = generatorWave(self.start, self.stop, self.center, self.delay, self.GD,
			self.GDD, self.TOD, self.FOD, self.QOD,
			self.resolution, self.delimiter, self.pulse_width, self.normalize, self.chirp)

	def show(self):
		"""
		Draws the plot of the generated data.
		"""
		self._check_norm()
		if np.iscomplexobj(self.y):
			self.plotwidget.plot(self.x, np.abs(self.y))
		else:   
			try:
				self.plotwidget.plot(self.x, self._y, 'r')
			except Exception:
				self.plotwidget.plot(self.x, self.y, 'r')
		self.plotwidget.grid()
		self.plotwidget.show()

	def save(self, name, path=None):
		"""
		Saves the generated dataset with numpy.savetxt.

		Parameters:
		----------

		name: string
		Name of the output file. You shouldn't include the .txt at the end.

		path: string, default is None
		You can also specify the save path.
		e.g path='C:/examplefolder'
		"""
		if path is None:
			np.savetxt('{}.txt'.format(name), np.transpose([self.x, self.y, self.ref, self.sam]), delimiter = self.delimiter)
			print('Successfully saved as {}'.format(name))
		else:
			np.savetxt(
				'{}/{}.txt'.format(path, name),
				 np.transpose([self.x, self.y, self.ref, self.sam]),
				 delimiter = self.delimiter
				 )
			print(f'Successfully saved as {name}')

	def _phase(self, j):
		if self.is_wave:
			lam = np.arange(self.start, self.stop + self.resolution, self.resolution) 
			omega = (2 * np.pi * C_LIGHT) / lam 
			omega0 = (2 * np.pi * C_LIGHT) / self.center 
			j = omega - omega0
		else:
			lamend = (2 * np.pi * C_LIGHT) / self.start
			lamstart = (2 * np.pi * C_LIGHT) / self.stop
			lam = np.arange(lamstart, lamend + self.resolution, self.resolution)
			omega = (2 * np.pi * C_LIGHT) / lam 
			j = omega - self.center
		return (j + self.delay * j + j * self.GD + (self.GDD / 2) * j ** 2 
			   + (self.TOD / 6)* j ** 3 + (self.FOD / 24) * j ** 4 + (self.QOD / 120) * j ** 5)

	def phase_graph(self):
		"""
		Plots the spectrogram along with the spectral phase.
		"""
		self._check_norm()
		self.fig, self.ax = self.plotwidget.subplots(2, 1, figsize=(8, 7))
		self.plotwidget.subplots_adjust(top=0.95)
		self.fig.canvas.set_window_title('Spectrum and phase')
		try:
			self.ax[0].plot(self.x, self._y, 'r')
		except Exception:
			self.ax[0].plot(self.x, self.y, 'r')
		try:
			self.ax[1].plot(self.x, self._phase(self.x))
		except Exception:
			raise ValueError('''The spectrum is not generated yet.
			Use self.generate_freq() on frequency domain or self.generate_wave() on wavelength domain.''')
		self.ax[0].set(xlabel="Frequency/Wavelength", ylabel="Intensity")
		self.ax[1].set(xlabel="Frequency/Wavelength", ylabel="$\Phi $[rad]")
		self.ax[0].grid()
		self.ax[1].grid()
		self.plotwidget.show()

	def unpack(self):
		"""
		Unpacks the generated data.
		If arms are given it returns x, y, reference_y, sample_y
		Else returns x, y
		"""
		if len(self.ref) == 0:
			return self.x, self.y
		return self.x, self.y, self.ref, self.sam


class Dataset(BaseApp):
	"""
	Base class for the evaluating methods.
	FIXME: ADD UNITS
	"""
	_metadata = None

	def __init__(self, x, y, ref=None, sam=None):
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
			except Exception:
				raise DatasetError('Invalid type of data')
		if not isinstance(self.y, np.ndarray):
			try:
				self.y = np.array(self.y)
				self.y.astype(float)
			except Exception:
				raise DatasetError('Invalid type of data')
		if not isinstance(self.ref, np.ndarray):
			try:
				self.ref = np.array(self.ref)
				self.ref.astype(float)
			except ValueError:
				pass
		if not isinstance(self.sam, np.ndarray):
			try:
				self.sam = np.array(self.sam)
				self.sam.astype(float)
			except ValueError:
				pass
		if len(self.ref) == 0:
			self.y_norm = self.y
		else:
			self.y_norm = (self.y - self.ref - self.sam) / (2 * np.sqrt(self.sam * self.ref))
			self._is_normalized = True
		self.plotwidget = plt
		self.xmin = None
		self.xmax = None
		self.probably_wavelength = None
		self._check_domain()

	def _check_domain(self):
		"""
		Checks the domain of data just by looking at x axis' minimal value.
		FIXME: Units are obviously not added yet, we work in nm and PHz...
		"""
		if min(self.x) > 100:
			self.probably_wavelength = True
		else:
			self.probably_wavelength = False

	@classmethod
	def parse_raw(cls, basefile, ref=None, sam=None, skiprows=8,
		decimal=',', sep=';'):
		'''
		Dataset object alternative constructor. Helps to load in data just by giving the filenames
		in the target directory.

		Parameters:
		----------
		basefile: str
		base interferogram
		*.trt file generated by the spectrometer

		ref: str, optional
		reference arm's spectra
		*.trt file generated by the spectrometer

		sam: str, optional
		sample arm's spectra
		*.trt file generated by the spectrometer
		'''
		with open(basefile) as file:
			cls._metadata = ''.join(next(file) for _ in range(4))
		df = pd.read_csv(basefile, skiprows=skiprows, sep=sep, decimal=decimal, names=['x', 'y'])
		if (ref and sam) is not None:
			r = pd.read_csv(ref, skiprows=skiprows, sep=sep, decimal=decimal, names=['x', 'y'])
			s = pd.read_csv(sam, skiprows=skiprows, sep = sep, decimal=decimal, names=['x', 'y'])
			return cls(df['x'].values, df['y'].values, r['y'].values, s['y'].values)
		return cls(df['x'].values, df['y'].values)

	def __str__(self):
		string = f'''
{type(self).__name__} object

Parameters
----------
Datapoints = {len(self.x)}
Normalized: {self._is_normalized}
Arms are separated: {True if len(self.ref) > 0 else False}
Predicted domain: {'wavelength' if self.probably_wavelength else 'frequency'}

Metadata extracted from file
----------------------------
{str(self._metadata)}
		'''
		return string

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

	def savgol_fil(self, window=101, order=3):
		"""
		Applies Savitzky-Golay filter on the dataset.

		Parameters:
		----------
		window: int, default is 101
		length of the convolutional window for the filter

		order: int, default is 3
		Degree of polynomial to fit after the convolution.
		If not odd, it's incremented by 1. Must be lower than window.
		Usually it's a good idea to stay with a low degree, e.g 3 or 5.

		Notes:
		------

		If arms were given, it will merge them into the self.y and self.y_norm variables.
		Also applies a linear interpolation on dataset (and raises warning).
		"""
		self.x, self.y_norm = savgol(self.x, self.y, self.ref, self.sam, window=window, order=order)
		self.y = self.y_norm
		self.ref = []
		self.sam = []
		warnings.warn('Linear interpolation have been applied to data.', InterpolationWarning)
		
	def slice(self, start=-9999, stop=9999):
		"""
		Cuts the dataset on x axis in this form: [start, stop]

		Parameters:
		----------
		start: float, default is -9999
		start value of cutting interval
		Not giving a value will keep the dataset's original minimum value.
		Note that giving -9999 will leave original minimum untouched too.

		stop: float, default is 9999
		stop value of cutting interval
		Not giving a value will keep the dataset's original maximum value.
		Note that giving 9999 will leave original maximum untouched too.

		Notes:
		------

		If arms were given, it will merge them into the self.y and self.y_norm variables.
		"""
		self.x, self.y_norm = cut_data(self.x, self.y, self.ref, self.sam, startValue=start, endValue=stop)
		self.ref = []
		self.sam = []
		self.y = self.y_norm
		# Just to make sure it's correctly shaped. Later on we might delete this.
		if type(self).__name__ == 'FFTMethod':
			self.original_x = self.x

	def convolution(self, window_length, std=20):
		"""
		Applies a convolution with a gaussian on the dataset

		Parameters:
		----------
		window_length: int
		length of the gaussian window

		std: float, default is 20
		standard deviation of the gaussian

		Notes:
		------

		If arms were given, it will merge them into the self.y and self.y_norm variables.
		Also applies a linear interpolation on dataset (and raises warning).
		"""
		self.x, self.y_norm = convolution(self.x, self.y, self.ref, self.sam, window_length, standev=std)
		self.ref = []
		self.sam = []
		self.y = self.y_norm
		warnings.warn('Linear interpolation have been applied to data.', InterpolationWarning)


	def detect_peak(self, pmax=0.1, pmin=0.1, threshold=0.1, except_around=None):
		"""
		Basic algorithm to find extremal points in data.

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

		Returns:
		-------
		xmax: array-like
		x coordinates of the maximums

		ymax: array-like
		y coordinates of the maximums

		xmin: array-like
		x coordinates of the minimums

		ymin: array-like
		y coordinates of the minimums
		"""
		xmax, ymax, xmin, ymin = find_peak(self.x, self.y, self.ref, self.sam, proMax=pmax, proMin=pmin, threshold=threshold, except_around=except_around)
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


class MinMaxMethod(Dataset):
	"""
	Basic interface for Minimum-Maximum Method.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def init_edit_session(self, pmax=0.1, pmin=0.1, threshold=0, except_around=None):
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
		
		For the time being, zooming will not persist, because upon modifying the points we always
		redraw the plot. Also, zooming will add a new point. We should rethink this and we might find 
		a better way later on.

		Currently this function is disabled when running it from IPython.
		"""
		if run_from_ipython():
			return '''It seems you run this code in IPython.
			Interactive plotting is not yet supported. 
			Consider running it in the regular console.'''
		_x, _y, _xx, _yy = self.detect_peak(
			pmax=pmax, pmin=pmin, threshold=threshold, except_around=except_around
			)
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
		print(f'In total {len(_editpeak.get_dat[0])} extremal points were recorded. Ready to calculate.')
		return _editpeak.get_dat[0]

	@print_disp
	def calculate(self, reference_point, fit_order, show_graph=False):
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
			maxx=self.xmax, minx=self.xmin, fitOrder=fit_order, showGraph=show_graph
			)
		return dispersion, dispersion_std, fit_report


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

	def smart_guess(self, reference_point=2.355, pmax=0.5, pmin=0.5, threshold=0.35):
		x_min, _, x_max, _ = self.detect_peak(pmax=pmax, pmin=pmin, threshold=threshold)
		try:
			closest_val, idx1 = find_nearest(x_min, reference_point)
			m_closest_val, m_idx1 = find_nearest(x_max, reference_point)
		except ValueError:
			print('No extremal values found. Adjust pmax, pmin, threshold parameters to find points.\nSkipping.. ')
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
			self.plotwidget.plot(self.x, self.fit, 'k--', label = 'fit', zorder=99)
			self.plotwidget.legend()
			self.show()
		else:
			self.show()

	def optimizer(self, reference_point, max_order=3, initial_region_ratio=0.1,
		extend_by=0.1, coef_threshold=0.3, max_tries=5000, show_endpoint=True):
		"""
		Cosine fit optimizer. It's based on adding new terms to fit function successively
		until we reach the max_order. 
		"""
		self.f = FitOptimizer(self.x, self.y, self.ref, self.sam, reference_point=reference_point,
		max_order=max_order)
		self.f.set_initial_region(initial_region_ratio)
		self.f.set_final_guess(GD=self.params[3], GDD=self.params[4], TOD=self.params[5],
		FOD=self.params[6], QOD=self.params[7]) # we can pass it higher params safely, they are ignored.
		self.f.run_loop(extend_by, coef_threshold, max_tries=max_tries, show_endpoint=show_endpoint)
		del self.f # making sure we don't subtract reference point again

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
		print('''With SPP-Method x and y values have a different meaning compared to other methods.
		Make sure you put delays to y and frequencies to x:
		SPPMethod(frequencies, delays)''')

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
	def calculate(self, reference_point, fit_order):
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
				self.y, self.x, reference_point=reference_point, fitOrder=fit_order, from_raw=True
				)
			self.om = self.x
			self.de = self.y
		else:
			self.om, self.de, dispersion, dispersion_std, self.bf = spp_method(
				self.y, self.x, fitOrder=fit_order, from_raw=False
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

	def window(self, at, std, window_order=6):
		"""
		Draws a gaussian window on the plot with the desired parameters.
		The maximum value is adjusted for the dataset mostly for visibility reasons.
		You should explicitly call self.show() after this function is set.

		Parameters:
		----------

		at: float
		maximum of the gaussian curve

		std: float #FIXME: RENAME THIS TO FWHM
		Full width at half maximum of the gaussian

		window_order: int, default is 6
		Order of the gaussian curve.
		If not even, it's incremented by 1 for safety reasons.
		"""
		self.at = at
		self.std = std
		self.window_order = window_order
		gaussian = gaussian_window(self.x, self.at, self.std, self.window_order)
		self.plotwidget.plot(self.x, gaussian*max(abs(self.y)), 'r--')

	def apply_window(self):
		"""
		If window function is correctly set, applies changes to the dataset.
		"""
		self.plotwidget.clf()
		self.plotwidget.cla()
		self.plotwidget.close()
		self.y = cut_gaussian(self.x, self.y, spike=self.at, sigma=self.std, win_order=self.window_order)
		
	@print_disp
	def calculate(self, reference_point, fit_order, show_graph=False):
		""" 
		FFTMethod's calculate function.

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

		Currently the x-axis transformation is sloppy, because we cache the original x axis and not transforming it
		backwards. In addition we need to keep track of interpolation and zero-padding too.
		Currently the transforms are correct only if first ifft was used.
		For now it's doing okay: giving good results. 
		For consistency we should still implement that a better way later.
		"""
		dispersion, dispersion_std, fit_report = args_comp(
			self.x, self.y, reference_point=reference_point, fitOrder=fit_order, showGraph=show_graph
			)
		return dispersion, dispersion_std, fit_report
