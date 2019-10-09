# -*- coding: utf-8 -*-

from math import factorial
import operator

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

try:
	from lmfit import Model
	_has_lmfit = True
except ImportError:
	_has_lmfit = False


__all__ = ['min_max_method', 'findNearest', 'cos_fit1', 'cos_fit2', 'cos_fit3',
		   'cos_fit4', 'cos_fit5', 'spp_method', 'cff_method', 'fft_method', 'cut_gaussian',
		   'ifft_method', 'args_comp']

def _handle_input(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY):
	"""
	Instead of handling the inputs in every function, there is this private method.

	Parameters
	----------

	initSpectrumX: array-like
	x-axis data

	initSpectrumY: array-like
	y-axis data

	referenceArmY, sampleArmY: array-like
	reference and sample arm spectrum evaluated at initSpectrumX

	Returns
	-------
	initSpectrumX: array-like
	unchanged x data

	Ydata: array-like
	the transformed y data

	"""
	if (len(initSpectrumX) > 0) and (len(referenceArmY) > 0) and (len(sampleArmY) > 0):
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
	elif (len(referenceArmY) == 0) or (len(sampleArmY) == 0):
		Ydata = initSpectrumY
	elif len(initSpectrumX) == 0:
		raise ValueError('Please load the spectrum!\n')
	else:
		raise TypeError('Input types are wrong.\n')
	return initSpectrumX,  Ydata


def min_max_method(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, ref_point, maxx=[], minx=[], fitOrder=5, showGraph=False):
	"""
	Calculates the dispersion with minimum-maximum method 

	Parameters
	----------

	initSpectrumX: array-like
	x-axis data

	initSpectrumY: array-like
	y-axis data

	referenceArmY, sampleArmY: array-like
	reference and sample arm spectra evaluated at initSpectrumX

	ref_point: float
	the reference point to calculate order
	
	maxx and minx: array-like
	the accepted minimal and maximal places (usually received from other methods)

	fitOrder: int
	degree of polynomial to fit data [1, 5]

	showGraph: bool
	if True returns a matplotlib plot and pauses execution until closing the window

	Returns
	-------

	dispersion: array-like
	[GD, GDD, TOD, FOD, QOD]

	dispersion_std: array-like
	[GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

	fit_report: lmfit report object
	
	"""
	Xdata, Ydata  = _handle_input(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY)

	_, SSPindex = findNearest(Xdata,ref_point)
	if len(maxx) == 0 or len(minx) == 0:
		maxInd = argrelextrema(Ydata, np.greater)
		minInd = argrelextrema(Ydata, np.less)
		maxx = Xdata[maxInd]
		minx = Xdata[minInd]
	else:
		maxx = maxx
		minx = minx

	relNegMaxFreqs = np.array([a for a in (Xdata[SSPindex]-maxx) if a<0])
	relNegMinFreqs= np.array([b for b in (Xdata[SSPindex]-minx) if b<0])
	relNegFreqs = relNegMaxFreqs
	relNegFreqs = sorted(np.append(relNegFreqs, relNegMinFreqs))
	relNegFreqs = relNegFreqs[::-1]
	relPosMaxFreqs = np.array([c for c in (Xdata[SSPindex]-maxx) if c>0])
	relPosMinFreqs= np.array([d for d in (Xdata[SSPindex]-minx) if d>0])
	relPosFreqs = relPosMinFreqs
	relPosFreqs = sorted(np.append(relPosFreqs,relPosMaxFreqs))

	negValues = np.zeros_like(relNegFreqs)
	posValues = np.zeros_like(relPosFreqs)
	for freq in range(len(relPosFreqs)):
		posValues[freq] = np.pi*(freq+1)
	for freq in range(len(relNegFreqs)):
		negValues[freq] = np.pi*(freq+1)
	x_s = np.append(relPosFreqs, relNegFreqs) 
	y_s = np.append(posValues, negValues)

	#making sure the data in right order
	L = sorted(zip(x_s,y_s), key=operator.itemgetter(0))
	fullXValues, fullYValues = zip(*L)
	
	if _has_lmfit:
		if fitOrder == 5:
			fitModel = Model(polynomialFit5)
			params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1, b4 = 1, b5 = 1)
			result = fitModel.fit(fullYValues, x=fullXValues, params = params, method ='leastsq') #nelder
		elif fitOrder == 4:
			fitModel = Model(polynomialFit4)
			params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1, b4 = 1)
			result = fitModel.fit(fullYValues, x=fullXValues, params = params, method ='leastsq') 
		elif fitOrder == 3:
			fitModel = Model(polynomialFit3)
			params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1)
			result = fitModel.fit(fullYValues, x=fullXValues, params = params, method ='leastsq') 
		elif fitOrder == 2:
			fitModel = Model(polynomialFit2)
			params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1)
			result = fitModel.fit(fullYValues, x=fullXValues, params = params, method ='leastsq') 
		elif fitOrder == 1:
			fitModel = Model(polynomialFit1)
			params = fitModel.make_params(b0 = 0, b1 = 1)
			result = fitModel.fit(fullYValues, x=fullXValues, params = params, method ='leastsq') 
		else:
			raise ValueError('Order is out of range, please select from [1,5]')
	else:
		if fitOrder == 5:
			popt, pcov = curve_fit(polynomialFit5, fullXValues, fullYValues, maxfev = 8000)
			_function = polynomialFit5
		elif fitOrder == 4:
			popt, pcov = curve_fit(polynomialFit4, fullXValues, fullYValues, maxfev = 8000)
			_function = polynomialFit4
		elif fitOrder == 3:
			popt, pcov = curve_fit(polynomialFit3, fullXValues, fullYValues, maxfev = 8000)
			_function = polynomialFit3
		elif fitOrder == 2:
			popt, pcov = curve_fit(polynomialFit2, fullXValues, fullYValues, maxfev = 8000)
			_function = polynomialFit2
		elif fitOrder == 1:
			popt, pcov = curve_fit(polynomialFit1, fullXValues, fullYValues, maxfev = 8000)
			_function = polynomialFit1
		else:
			raise ValueError('Order is out of range, please select from [1,5]')
	try:
		if _has_lmfit:
			dispersion = []
			dispersion_std = []
			for name, par in result.params.items():
				dispersion.append(par.value)
				dispersion_std.append(par.stderr)
			dispersion = dispersion[1:]
			dispersion_std = dispersion_std[1:]
			for idx in range(len(dispersion)):
				dispersion[idx] =  dispersion[idx] / factorial(idx+1) 
				dispersion_std[idx] =  dispersion_std[idx] / factorial(idx+1)
			while len(dispersion)<5:
				dispersion.append(0)
				dispersion_std.append(0) 
			fit_report = result.fit_report()
		else:
			fullXValues = np.asarray(fullXValues)
			dispersion=[]
			dispersion_std=[]
			for idx in range(len(popt)-1):
				dispersion.append(popt[idx+1]/factorial(idx+1))
			while len(dispersion)<5:
				dispersion.append(0)
			while len(dispersion_std)<len(dispersion):
				dispersion_std.append(0)
			fit_report = '\nTo display detailed results, you must have lmfit installed.'
		if showGraph:
			fig = plt.figure(figsize=(7,7))
			fig.canvas.set_window_title('Min-max method fitted')
			plt.plot(fullXValues, fullYValues, 'o', label = 'dataset')
			try:
				plt.plot(fullXValues, result.best_fit, 'r*', label = 'fitted')
			except:
				plt.plot(fullXValues, _function(fullXValues, *popt), 'r*', label = 'fitted')
			plt.legend()
			plt.grid()
			plt.show()
		return dispersion, dispersion_std, fit_report
	except Exception as e:
		return [],[],e	


def polynomialFit5(x, b0, b1, b2, b3, b4, b5):
	"""
	Taylor polynomial for fit
	b1 = GD
	b2 = GDD / 2
	b3 = TOD / 6
	b4 = FOD / 24
	b5 = QOD / 120
	"""
	return b0+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5

def polynomialFit4(x, b0, b1, b2, b3, b4):
	"""
	Taylor polynomial for fit
	b1 = GD
	b2 = GDD / 2
	b3 = TOD / 6
	b4 = FOD / 24
	"""
	return b0+b1*x+b2*x**2+b3*x**3+b4*x**4

def polynomialFit3(x, b0, b1, b2, b3):
	"""
	Taylor polynomial for fit
	b1 = GD
	b2 = GDD / 2
	b3 = TOD / 6

	"""
	return b0+b1*x+b2*x**2+b3*x**3

def polynomialFit2(x, b0, b1, b2):
	"""
	Taylor polynomial for fit
	b1 = GD
	b2 = GDD / 2
	"""
	return b0+b1*x+b2*x**2

def polynomialFit1(x, b0, b1):
	"""
	Taylor polynomial for fit
	b1 = GD
	"""
	return b0+b1*x



def findNearest(array, value):
	"""
	Finds the nearest element to the given value in the array
	"""
	array = np.asarray(array)
	idx = (np.abs(value-array)).argmin()
	return array[idx], idx

def cos_fit1(x,c0, c1, b0, b1):
	return c0 + c1*np.cos(b0 + b1*x)

def cos_fit2(x,c0, c1, b0, b1, b2):
	return c0 + c1*np.cos(b0 + b1*x + b2*x**2)

def cos_fit4(x,c0, c1, b0, b1, b2, b3, b4):
	return c0 + c1*np.cos(b0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4)

def cos_fit5(x,c0, c1, b0, b1, b2, b3, b4, b5):
	return c0 + c1*np.cos(b0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4 + b5*x**5)

def cos_fit3(x,c0, c1, b0, b1, b2, b3):
	return c0 + c1*np.cos(b0 + b1*x + b2*x**2 + b3*x**3)



def spp_method(delays, omegas, fitOrder=4, from_raw=False): 
	"""
	Calculates the dispersion from SPP's positions and delays.
	
	Parameters
	----------

	delays: array-like
	the time delays in fs
	if from_raw is enabled you must pass matching pairs with omegas

	omegas: array-like
	in form of [[SPP1, SPP2, SPP3, SPP4],[SPP1, SPP2, SPP3, SPP4], ..]
	for lesser SPP cases replace elements with None:
	[[SPP1, None, None, None],[SPP1, None, None, None], ..]
	if from_raw is enabled, you must pass matching pairs with delays

	fitOrder: int
	order of polynomial to fit the given data

	from_raw: bool
	if True you can pass matching pairs to delays and omegas, and it will perform 
	a normal curve fitting. It's useful at the API.

	Returns
	-------
	omegas_unpacked: array-like
	x axis data

	delays_unpacked : array-like
	y axis data

	dispersion: array-like
	[GD, GDD, TOD, FOD, QOD]

	dispersion_std: array-like
	[GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

	bf: array-like
	best fitting curve for plotting
	"""
	if from_raw:
		delays_unpacked = delays
		omegas_unpacked = omegas
	else:
		delays = delays[delays != np.array(None)]
		omegas_unpacked = []
		delays_unpacked = []
		for delay, element in zip(delays, omegas):
			item = [x for x in element if x is not None]
			omegas_unpacked.extend(item)
			delays_unpacked.extend(len(item) * [delay])
	try:
		if _has_lmfit:
			if fitOrder == 2:
				fitModel = Model(polynomialFit2)
				params = fitModel.make_params(b0 = 1, b1 = 1, b2 = 1)
				result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
			elif fitOrder == 3:
				fitModel = Model(polynomialFit3)
				params = fitModel.make_params(b0 = 1, b1 = 1, b2 = 1, b3 = 3)
				result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
			elif fitOrder == 4:
				fitModel = Model(polynomialFit4)
				params = fitModel.make_params(b0 = 1, b1 = 1, b2 = 1, b3=1, b4 =1)
				result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
			elif fitOrder == 1:
				fitModel = Model(polynomialFit1)
				params = fitModel.make_params(b0 = 1, b1 = 1)
				result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
			else:
				raise ValueError('Order is out of range, please select from [1,4]')
			dispersion = []
			dispersion_std = []
			for name, par in result.params.items():
				dispersion.append(par.value)
				dispersion_std.append(par.stderr)
			for idx in range(len(dispersion)):
				dispersion[idx] =  dispersion[idx]*factorial(idx) #biztos?
				dispersion_std[idx] =  dispersion_std[idx] * factorial(idx)
			while len(dispersion)<5:
				dispersion.append(0)
				dispersion_std.append(0) 
			bf = result.best_fit
		else:
			if fitOrder == 4:
				popt, pcov = curve_fit(polynomialFit4, omegas_unpacked, delays_unpacked, maxfev = 8000)
				_function = polynomialFit4
			elif fitOrder == 3:
				popt, pcov = curve_fit(polynomialFit3, omegas_unpacked, delays_unpacked, maxfev = 8000)
				_function = polynomialFit3
			elif fitOrder == 2:
				popt, pcov = curve_fit(polynomialFit2, omegas_unpacked, delays_unpacked, maxfev = 8000)
				_function = polynomialFit2
			elif fitOrder == 1:
				popt, pcov = curve_fit(polynomialFit1, omegas_unpacked, delays_unpacked, maxfev = 8000)
				_function = polynomialFit1
			else:
				raise ValueError('Order is out of range, please select from [1,4]')
			#FIXME: biztos?
			omegas_unpacked = np.asarray(omegas_unpacked)
			dispersion=[]
			dispersion_std=[]
			for idx in range(len(popt)):
				dispersion.append(popt[idx]*factorial(idx))
			while len(dispersion)<5:
				dispersion.append(0)
			while len(dispersion_std)<len(dispersion):
				dispersion_std.append(0)
			bf = _function(omegas_unpacked, *popt)
		return omegas_unpacked, delays_unpacked, dispersion, dispersion_std, bf
	except Exception as e:
		return [], [], [e], [], []


def cff_method(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, ref_point=0 , p0=[1, 1, 1, 1, 1, 1, 1, 1]):
	"""
	Phase modulated cosine function fit method. 
	

	Parameters
	----------
	
	initSpectrumX: array-like
	x-axis data

	initSpectrumY: array-like
	y-axis data

	referenceArmY, sampleArmY: array-like
	the reference and sample arm spectra evaluated at initSpectrumX

	p0: array-like
	the initial parameters for fitting

	Returns
	-------

	dispersion: array-like
	[GD, GDD, TOD, FOD, QOD]

	bf: array-like
	best fitting curve

	"""
	# TODO: BOUNDS WILL BE SET  ..
	# bounds=((-1000, -10000, -10000, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), 
		    # (1000, 10000, 10000, np.inf, np.inf, np.inf, np.inf, np.inf))
	Xdata, Ydata = _handle_input(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY)
	#TODO: replace with lmfit
	try:
		if p0[-1] == 0 and p0[-2] == 0 and p0[-3] == 0 and p0[-4] == 0:
			_funct = cos_fit1
			p0 = p0[:-4]
		elif p0[-1] == 0 and p0[-2] == 0 and p0[-3] == 0:
			_funct = cos_fit2
			p0 = p0[:-3]
		elif p0[-1] == 0 and p0[-2] == 0:
			_funct = cos_fit3
			p0 = p0[:-2]
		elif p0[-1] == 0:
			_funct = cos_fit4
			p0 = p0[:-1]
		else:
			_funct = cos_fit5
		popt, pcov = curve_fit(_funct, Xdata-ref_point, Ydata, p0, maxfev = 8000)
		dispersion = np.zeros_like(popt)[:-3]
		for num in range(len(popt)-3):
			dispersion[num] = popt[num+3]/factorial(num)
		# fig1 = plt.figure()
		# fig1.canvas.set_window_title('Cosine function fit method')
		# plt.plot(Xdata, Ydata,'r-',label = 'dataset')
		# plt.plot(Xdata, cosFitForPMCFF(Xdata, *popt),'k*', label = 'fitted')
		# plt.legend()
		# plt.grid()
		# plt.show()
		return dispersion, _funct(Xdata-ref_point, *popt)
	except RuntimeError:
		raise ValueError('Max tries reached.. \n Parameters could not be estimated.')


def fft_method(initSpectrumY):
	"""
	Perfoms FFT on data

	Parameters
	----------

	initSpectrumY: array-like
	y-axis data
	
	Returns
	-------

	yf: array-like
	the transformed y data

	"""
	if len(initSpectrumY) > 0:
		Ydata = initSpectrumY
		yf = scipy.fftpack.fft(Ydata)
		return yf
	if len(initSpectrumY) == 0:
		pass

def gaussian_window(t ,tau, standardDev, order):
	"""
	Parameters
	----------
	t: array-like
	input array to perform window on

	tau: float
	center of gaussian window

	standardDev: float
	standard deviation of gaussian window

	Returns
	-------
	arr : array-like
	6th order gaussian window with params above

	"""
	return np.exp(-(t-tau)**order/(2*standardDev**order))

def cut_gaussian(initSpectrumX, initSpectrumY, spike, sigma, win_order):
	"""
	Applies gaussian window with the given params.

	Parameters
	----------
	initSpectrumX: array-like
	x-axis data

	initSpectrumY: array-like
	y-axis data

	spike: float
	center of gaussian window

	sigma: float
	standard deviation of gaussian window

	Returns
	-------

	Ydata: array-like
	the windowed y values 
	
	"""

	Ydata = initSpectrumY * gaussian_window(initSpectrumX, tau = spike, standardDev = sigma, order = win_order) 
	# Ydata = initSpectrumY * scipy.signal.windows.gaussian(len(initSpectrumY), std=sigma)
	return Ydata



def ifft_method(initSpectrumX, initSpectrumY, interpolate = True):
	"""
	Perfoms IFFT on data

	Parameters
	----------

	initSpectrumX: array-like
	the x-axis data

	initSpectrumY: array-like
	the y-axis data

	interpolate: bool
	if True perform a linear interpolation on dataset before transforms
	
	Returns
	-------
	freq: array-like
	the transformed x data

	yf: array-like
	transformed y data

	"""
	from .edit_features import interpolate_data
	if len(initSpectrumY)>0 and len(initSpectrumX)>0:
		Ydata = initSpectrumY
		Xdata = initSpectrumX
	else:
		raise
	if interpolate:
		Xdata, Ydata = interpolate_data(initSpectrumX, initSpectrumY, [],[])
	yf = scipy.fftpack.ifft(Ydata)
	freq = scipy.fftpack.fftfreq(len(Xdata), d=(Xdata[3]-Xdata[2]))
	return freq, yf 
	



def args_comp(initSpectrumX, initSpectrumY, fitOrder=5, showGraph=False):
	"""
	Calculates the phase of complex dataset then unwrap by changing deltas between 
	values to 2*pi complement. At the end, fit a polynomial curve to determine
	dispersion coeffs.

	Parameters
	----------

	initSpectrumX: array-like
	the x-axis data

	initSpectrumY: array-like
	the y-axis data

	fitOrder: int
	degree of polynomial to fit data [1, 5]

	showGrap: bool
	if True returns a matplotlib plot and pauses execution until closing the window
	
	Returns
	-------

	dispersion: array-like
	[GD, GDD, TOD, FOD, QOD]

	dispersion_std: array-like
	[GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

	fit_report: lmfit report object

	"""
	angles = np.angle(initSpectrumY)
	###shifting to [0, 2pi]
	# angles = (angles + 2 * np.pi) % (2 * np.pi)
	Xdata = initSpectrumX
	Ydata = np.unwrap(angles, axis = 0)
	if fitOrder == 5:
		fitModel = Model(polynomialFit5)
		params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1, b4 = 1, b5 = 1)
		result = fitModel.fit(Ydata, x=Xdata, params = params, method ='leastsq') 
	elif fitOrder == 4:
		fitModel = Model(polynomialFit4)
		params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1, b4 = 1)
		result = fitModel.fit(Ydata, x=Xdata, params = params, method ='leastsq') 
	elif fitOrder == 3:
		fitModel = Model(polynomialFit3)
		params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1)
		result = fitModel.fit(Ydata, x=Xdata, params = params, method ='leastsq') 
	elif fitOrder == 2:
		fitModel = Model(polynomialFit2)
		params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1)
		result = fitModel.fit(Ydata, x=Xdata, params = params, method ='leastsq') 
	elif fitOrder == 1:
		fitModel = Model(polynomialFit1)
		params = fitModel.make_params(b0 = 0, b1 = 1)
		result = fitModel.fit(Ydata, x=Xdata, params = params, method ='leastsq') 
	else:
		raise ValueError('Order is out of range, please select from [1,5]')
	try:
		dispersion = []
		dispersion_std = []
		for name, par in result.params.items():
			dispersion.append(par.value)
			dispersion_std.append(par.stderr)
		dispersion = dispersion[1:]
		dispersion_std = dispersion_std[1:]
		for idx in range(len(dispersion)):
			dispersion[idx] =  dispersion[idx] / factorial(idx+1) 
			dispersion_std[idx] =  dispersion_std[idx] / factorial(idx+1)
		while len(dispersion)<5:
			dispersion.append(0)
			dispersion_std.append(0) 
		fit_report = result.fit_report()
		if showGraph:
			fig = plt.figure(figsize=(7,7))
			fig.canvas.set_window_title('Phase')
			plt.plot(Xdata, Ydata, 'o', label = 'dataset')
			plt.plot(Xdata, result.best_fit, 'r--', label = 'fitted')
			plt.legend()
			plt.grid()
			plt.show()
		return dispersion, dispersion_std, fit_report
	except Exception as e:
		return [],[],e



""" #Teszt
a, b = np.loadtxt('ff.txt', unpack = True, delimiter = ',')
aa, bb = ifft_method(a, b, interpolate = True)
bbb = cut_gaussian(aa, bb, 740, 100, 6) 
bbbb = fft_method(bbb)
angles = np.angle(bbbb)
angles = (angles + 2 * np.pi) % (2 * np.pi)
c = np.unwrap(angles)
plt.plot(aa, c, 'ro')
plt.show()
"""