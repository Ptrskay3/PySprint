# -*- coding: utf-8 -*-
##############################
#
#
# Rewriting comes next.
#
#
##############################

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import scipy
from math import factorial
from lmfit import Model
from smoothing import savgolFilter, findPeaks, convolution, interpolateData, cutData, findNearest


#majd új argumentumként a mértékegységet bele kell vinni, ez a legjobb megoldás talán

def minMaxMethod(initSpectrumX, initSpectrumY, referenceArmY , sampleArmY, SPPosition, maxx=[], minx=[]):
	"""
	Minimum-maximum method
	Takes in the interferogram with the reference and sample arm spectra, and also the SPP Position as argument.
	The initSpectrumX defaults to angular frequency in PHz.
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
	#ide jön majd a zajszűrés, min maxok kiválasztása valahogyan
	SSPinData, SSPindex = findNearest(Xdata, SPPosition)
	if len(maxx) == 0 or len(minx) == 0:
		maxInd = argrelextrema(Ydata, np.greater)
		minInd = argrelextrema(Ydata, np.less)
		maxx = Xdata[maxInd]
		minx = Xdata[minInd]
	else:
		maxx = maxx
		minx = minx
	#######
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
	
	try:
		fitModel = Model(polynomialFit)
		params = fitModel.make_params(b0 = 0, b1 = 1, b2 = 1, b3 = 1, b4 = 1, b5 = 1)
		result = fitModel.fit(fullYValues, x=fullXValues, params = params, method ='nelder')
		dispersion = []
		dispersion_std = []
		for name, par in result.params.items():
			dispersion.append(par.value)
			dispersion_std.append(par.stderr)
		dispersion = dispersion[1:]
		dispersion_std = dispersion_std[1:]
		for idx in range(len(dispersion)):
			dispersion[idx] = factorial(idx+1) * dispersion[idx]
			dispersion_std[idx] = factorial(idx+1) * dispersion_std[idx]
		fit_report = result.fit_report()
		# popt, pcov = curve_fit(polynomialFit, fullXValues, fullYValues)
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
		return dispersion, dispersion_std, fit_report
	except:
		return ['Optimal','parameters', 'not', 'found', '.'], [], []




def polynomialFit(x, b0, b1, b2, b3, b4, b5):
	#Helper function 
	return b0+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5


def findNearest(array, value):
	#Finds the nearest element to the given value in the array
	#returns tuple: (element, element's index)
	
    array = np.asarray(array)
    idx = (np.abs(value - array)).argmin()
    return array[idx], idx


def cosFitForPMCFF(x,c0, c1, b0, b1, b2, b3, b4, b5):
	"""
	Helper function for Phase Modulated Cosine Function Fit 
	"""
	return c0 + c1*np.cos(b0+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5)


def SSP():
	pass


def PMCFFMethod(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, p0=[1, 1, 1, 1, 1, 1, 1, 1]):
	"""
	Phase modulated cosine function fit method. p0 is the array containing inital parameters for fitting.
	initSpectrumX default to angular frequency
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

	try:
		popt, pcov = curve_fit(cosFitForPMCFF, Xdata, Ydata, p0, maxfev = 5000, bounds = bounds)
		dispersion = np.zeros_like(popt)[:-3]
		for num in range(len(popt)-3):
			dispersion[num] = popt[num+3]*factorial(num)
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
	if len(initSpectrumX) > 0 and len(initSpectrumY) > 0:
		Xdata, Ydata = interpolateData(initSpectrumX, initSpectrumY, [],[])
		freq = scipy.fftpack.fftfreq(len(Xdata), d=(Xdata[3]-Xdata[2]))
		yf = scipy.fftpack.fft(Ydata)
		return freq, yf
	if len(initSpectrumX) == 0:
		pass

def gaussianWindow(t ,tau, standardDev):
	return np.exp(-(t-tau)**6/(2*standardDev**6))

def cutWithGaussian(initSpectrumX ,initSpectrumY, spike, sigma):
	Ydata = initSpectrumY * gaussianWindow(initSpectrumX, tau = spike, standardDev=sigma) 
	# Ydata = initSpectrumY * scipy.signal.windows.gaussian(len(initSpectrumY), std=sigma)
	return Ydata


def IFFT(initSpectrumY):
	if len(initSpectrumY)>0:
		Ydata = initSpectrumY
		yf = scipy.fftpack.ifft(Ydata)
		return yf 
	else:
		pass

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


