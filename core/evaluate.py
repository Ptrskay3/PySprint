# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import scipy
from math import factorial
from lmfit import Model
from itertools import chain



# input unit can be added as new arg


def minMaxMethod(initSpectrumX, initSpectrumY, referenceArmY , sampleArmY, SPPosition, maxx=[], minx=[], fitOrder = 5, showGraph = False):
	"""
	Minimum-maximum method 
	(*CURRENTLY ACCEPTS UNITS ONLY IN PHz)

	__inputs__

	initSpectrumX:
	array with the x-axis data

	initSpectrumY:
	array with the y-axis data

	referenceArmY, sampleArmY:
	arrays containing the reference and sample arm spectra evaluated at initSpectrumX

	SPPosition:
	float, the reference point to calculate order
	
	maxx and minx:
	arrays containing the accepted minimal and maximal places (recieved from another functions)

	fitOrder:
	int, degree of polynomial to fit data

	showGraph:
	bool, if True returns a matplotlib plot and pauses execution until closing the window

	__returns__

	dispersion:
	array with shape and values:[GD, GDD, TOD, FOD, QOD]

	dispersion_std:
	array with the standard deviation for dispersion [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

	fit_report:
	lmfit report object
	"""

	if (len(initSpectrumX) > 0) and (len(referenceArmY) > 0) and (len(sampleArmY) > 0) and (SPPosition is not None):
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
	elif (len(referenceArmY) == 0) or (len(sampleArmY) == 0):
		Ydata = initSpectrumY
	elif len(initSpectrumX)== 0:
		raise ValueError('Please load the spectrum!\n')
	else:
		raise ValueError('Something went wrong...')

	Xdata = initSpectrumX

	_, SSPindex = findNearest(Xdata,SPPosition)
	if len(maxx) == 0 or len(minx) == 0:
		maxInd = argrelextrema(Ydata, np.greater)
		minInd = argrelextrema(Ydata, np.less)
		maxx = Xdata[maxInd]
		minx = Xdata[minInd]
	else:
		maxx = maxx
		minx = minx

	# plt.plot(Xdata, Ydata)
	# plt.plot(maxx, np.zeros_like(maxx), 'o')
	# plt.plot(minx, np.zeros_like(minx), 'o')
	# plt.show()

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
	fullXValues = np.append(relPosFreqs, relNegFreqs) 
	fullYValues = np.append(posValues, negValues)
	# fullXValues = np.append(fullXValues, 0)
	# fullYValues = np.append(fullYValues, 0)

	 
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
			fig.canvas.set_window_title('Min-max method fitted')
			plt.plot(fullXValues, fullYValues, 'o', label = 'dataset')
			plt.plot(fullXValues, result.best_fit, 'r*', label = 'fitted')
			plt.legend()
			plt.grid()
			plt.show()
		"""OLD
		# popt, pcov = curve_fit(polynomialFit5, fullXValues, fullYValues)
		# dispersion = np.zeros_like(popt)
		# for num in range(len(popt)):
		# 	dispersion[num] = popt[num]*factorial(num)
			# fig = plt.figure()
			# fig.canvas.set_window_title('Min-Max method')
			# plt.plot(fullXValues, fullYValues,'ro',label = 'dataset')
			# plt.plot(fullXValues, polynomialFit(fullXValues, *popt),'k*', label = 'fitted')
			# plt.legend()
			# plt.xlabel('$\Delta \omega or \Delta \lambda$')
			# plt.ylabel('Phase')
			# plt.grid()
			# plt.show()
		"""
		return dispersion, dispersion_std, fit_report
	except:
		return ['Optimal', 'params', 'not', 'found', '.'], ['','','','',''], []




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

""" Tesing
# a, b, c, d = np.loadtxt('examples/teszt.txt', unpack= True, delimiter=',')
# ds, dss, res=  minMaxMethod(a,b,c,d, 2.5)
"""


def findNearest(array, value):
	#Finds the nearest element to the given value in the array
	#returns tuple: (element, element's index)
	
    array = np.asarray(array)
    idx = (np.abs(value - array)).argmin()
    return array[idx], idx


def cosFitForPMCFF(x,c0, c1, b0, b1, b2, b3, b4, b5):
	"""
	Helper function for Phase Modulated Cosine Function Fit 
	b1 = GD
	b2 = GDD / 2
	b3 = TOD / 6
	b4 = FOD / 24
	b5 = QOD / 120
	"""
	return c0 + c1*np.cos(b0+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5)



def SPP(delays,omegas, fitOrder= 4): #def SPP(delays,omegas, reference, fitOrder= 4):
	extracted_omegas = []
	delays = delays[delays != np.array(None)]
	for element in omegas:
		item = [x for x in element if x is not None]
		extracted_omegas.append(item)

	new_delays = []
	for idx in range(len(extracted_omegas)):
		if len(extracted_omegas[idx])>1:
			value = delays[idx]
			item_to_add = [value]*(len(extracted_omegas[idx]))
			new_delays.append(item_to_add)
		if len(extracted_omegas[idx]) == 1:
			value = delays[idx]
			new_delays.append([value])

	delays_unpacked = []
	omegas_unpacked = []
	for element in extracted_omegas:
		for item in element:
			omegas_unpacked.append(item)
	delays_unpacked = list(chain.from_iterable(new_delays))

	# # omegas_unpacked = [x-reference for x in omegas_unpacked]

	# print(omegas_unpacked)
	# print(delays_unpacked)
	try:

		if fitOrder == 2:
			fitModel = Model(polynomialFit2)
			params = fitModel.make_params(b0 = 1, b1 = 1, b2 = 1)
			result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
		if fitOrder == 3:
			fitModel = Model(polynomialFit3)
			params = fitModel.make_params(b0 = 1, b1 = 1, b2 = 1, b3 = 3)
			result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
		if fitOrder == 4:
			fitModel = Model(polynomialFit4)
			params = fitModel.make_params(b0 = 1, b1 = 1, b2 = 1, b3=1, b4 =1)
			result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
		if fitOrder == 1:
			fitModel = Model(polynomialFit1)
			params = fitModel.make_params(b0 = 1, b1 = 1)
			result = fitModel.fit(delays_unpacked, x=omegas_unpacked, params = params, method ='leastsq') #nelder
		dispersion = []
		dispersion_std = []
		for name, par in result.params.items():
			dispersion.append(par.value)
			dispersion_std.append(par.stderr)
		for idx in range(len(dispersion)):
			dispersion[idx] =  dispersion[idx]*factorial(idx) 
			dispersion_std[idx] =  dispersion_std[idx] * factorial(idx)
		while len(dispersion)<5:
			dispersion.append(0)
			dispersion_std.append(0) 
		bf = result.best_fit
	except Exception as e:
		return e

	return omegas_unpacked, delays_unpacked, dispersion, dispersion_std, bf



def PMCFFMethod(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, p0=[1, 1, 1, 1, 1, 1, 1, 1]):
	"""
	Phase modulated cosine function fit method. 
	(*CURRENTLY ACCEPTS UNITS ONLY IN PHz)

	__inputs__
	
	initSpectrumX:
	array with the x-axis data

	initSpectrumY:
	array with the y-axis data

	referenceArmY, sampleArmY:
	arrays containing the reference and sample arm spectra evaluated at initSpectrumX

	p0:
	array with the initial parameters for fitting

	__returns__

	dispersion:
	array with shape and values:[GD, GDD, TOD, FOD, QOD]

	"""
	bounds=((-1, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (1, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf))
	if len(initSpectrumY) > 0 and len(referenceArmY) > 0 and len(sampleArmY) > 0:
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
		Ydata = np.asarray(Ydata)
	elif len(initSpectrumY) == 0:
		raise ValueError('Please load the spectrum!\n')
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = np.asarray(initSpectrumY)
	else:
		raise ValueError('No data..')
	Xdata = np.asarray(initSpectrumX)

	try: ## will be replaced with lmfit 
		popt, pcov = curve_fit(cosFitForPMCFF, Xdata, Ydata, p0, maxfev = 5000, bounds = bounds)
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
		return dispersion
	except RuntimeError:
		return ['Optimal','parameters', 'not', 'found', '.']



def FFT(initSpectrumX, initSpectrumY):
	"""
	Perfoms FFT on data

	__inputs__

	initSpectrumX:
	array with the x-axis data

	initSpectrumY:
	array with the y-axis data
	
	__returns__
	
	freq: 
	array with the transformed x axis

	yf:
	array with the transformed y data

	"""
	from .smoothing import interpolateData
	if len(initSpectrumX) > 0 and len(initSpectrumY) > 0:
		Xdata, Ydata = interpolateData(initSpectrumX, initSpectrumY, [],[])
		freq = scipy.fftpack.fftfreq(len(Xdata), d=(Xdata[3]-Xdata[2]))
		yf = scipy.fftpack.fft(Ydata)
		return freq, yf
	if len(initSpectrumX) == 0:
		pass

def gaussianWindow(t ,tau, standardDev):
	"""
	__inputs__
	t:
	input array

	tau :
	float, center of gaussian window

	standardDev:
	float, standard deviation of gaussian window

	__returns__
	6th order gaussian window with params above


	"""
	return np.exp(-(t-tau)**6/(2*standardDev**6))

def cutWithGaussian(initSpectrumX ,initSpectrumY, spike, sigma):
	"""
	Applies gaussian window with the given params.

	__inputs__

	initSpectrumX:
	array with the x-axis data

	initSpectrumY:
	array with the y-axis data

	spike:
	float, center of gaussian window

	sigma:
	float, standard deviation of gaussian window

	__returns__

	Ydata:
	array with windowed y values 
	
	"""

	Ydata = initSpectrumY * gaussianWindow(initSpectrumX, tau = spike, standardDev=sigma) 
	# Ydata = initSpectrumY * scipy.signal.windows.gaussian(len(initSpectrumY), std=sigma)
	return Ydata


def IFFT(initSpectrumY):
	"""
	Perfoms IFFT on data

	__inputs__

	initSpectrumY:
	array with the y-axis data
	
	__returns__

	yf:
	array with the transformed y data

	"""
	if len(initSpectrumY)>0:
		Ydata = initSpectrumY
		yf = scipy.fftpack.ifft(Ydata)
		return yf 
	else:
		pass

# under testing..
def argsAndCompute(initSpectrumX, initSpectrumY):
	angles = np.angle(initSpectrumY)
	Xdata = initSpectrumX
	popt, pcov = curve_fit(polynomialFit, Xdata, angles)
	fig1 = plt.figure()
	plt.plot(Xdata, angles,'ro',label = 'dataset')
	plt.plot(Xdata, polynomialFit(Xdata, *popt),'k*', label = 'fitted')
	plt.legend()
	plt.grid()
	plt.show()

"""#TESZT
a, b = np.loadtxt('examples/fft.txt', unpack = True, delimiter = ',')
aa, bb = FFT(a,b)
bbb = cutWithGaussian(aa, bb, spike = 223, sigma = 10)
bbbb = IFFT(bbb)
# plt.plot(a, bbbb)
plt.show()
argsAndCompute(a, bbbb)
"""


