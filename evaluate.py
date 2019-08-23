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
from smoothing import savgolFilter, findPeaks, convolution, interpolateData, cutData, findNearest


#majd új argumentumként a mértékegységet bele kell vinni, ez a legjobb megoldás talán

def minMaxMethod(initSpectrumX, initSpectrumY, referenceArmY , sampleArmY, SPPosition, showGraph=False):
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
	maxInd = argrelextrema(Ydata, np.greater)
	minInd = argrelextrema(Ydata, np.less)
	maxx = Xdata[maxInd]
	minx = Xdata[minInd]
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
	fullXValues = np.append(relPosFreqs, relNegFreqs) # ez itt nem biztos, hogy helyes
	fullYValues = np.append(posValues, negValues)
	
	try:
		popt, pcov = curve_fit(polynomialFitForMM, fullXValues, fullYValues)
		dispersion = np.zeros_like(popt)

		for num in range(len(popt)):
			dispersion[num] = popt[num]*factorial(num)
			if showGraph == True:	
				fig = plt.figure()
				fig.canvas.set_window_title('Min-Max method')
				plt.plot(fullXValues, fullYValues,'ro',label = 'dataset')
				plt.plot(fullXValues, polynomialFitForMM(fullXValues, *popt),'k*', label = 'fitted')
				plt.legend()
				plt.xlabel('$\Delta \omega or \Delta \lambda$')
				plt.ylabel('Phase')
				plt.grid()
				plt.show()
			else:
				pass
		return dispersion
	except Exception as e:
		return e




def polynomialFitForMM(x, b0, b1, b2, b3, b4, b5):
	#Helper function for Min-Max Method
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
############################################################################
#
#  A RELATÍV KÖRFREKVENCIÁRA KELL ILLESZTENI, ÁT KELL DOLGOZNI.
#
############################################################################
def PMCFFMethod(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, p0=[1, 1, 1, 1, 1, 1, 1, 1], showGraph = False):
	"""
	Phase modulated cosine function fit method. p0 is the array containing inital parameters for fitting.
	initSpectrumX default to angular frequency
	"""
	if len(initSpectrumY) > 0 and len(referenceArmY) > 0 and len(sampleArmY) > 0:
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
	elif len(initSpectrumY) == 0:
		raise ValueError('Please load the spectrum!\n')
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = initSpectrumY
	else:
		raise RuntimeError('Operation timed out. Try again!\n')
	Xdata = initSpectrumX
	try:
		popt, pcov = curve_fit(cosFitForPMCFF, Xdata, Ydata, p0)
		dispersion = np.zeros_like(popt)[:-3]
		for num in range(len(popt)-3):
			dispersion[num] = popt[num+3]*factorial(num)
		if showGraph == True:
			fig1 = plt.figure()
			fig1.canvas.set_window_title('Cosine function fit method')
			plt.plot(Xdata, Ydata,'ro',label = 'dataset')
			plt.plot(Xdata, cosFitForPMCFF(Xdata, *popt),'k*', label = 'fitted')
			plt.legend()
			plt.xlabel('$\Delta \omega or \Delta \lambda$')
			plt.ylabel('Phase')
			plt.grid()
			plt.show()
		else:
			pass
		return dispersion
	except Exception as e:
		return e


def FFT(initSpectrumX, initSpectrumY):
	if len(initSpectrumX) > 0 and len(initSpectrumY) > 0:
		Xdata, Ydata = interpolateData(initSpectrumX, initSpectrumY, [],[])
		# Xdata = initSpectrumX
		# Ydata = initSpectrumY
		freq = scipy.fftpack.fftfreq(len(Xdata), d=(Xdata[3]-Xdata[2]))
		yf = scipy.fftpack.fft(Ydata)
		# plt.figure()
		# lol = plt.plot(freq, np.abs(yf), 'r', markersize = 4)
		# plt.grid()
		# plt.show()
		return freq, yf
	if len(initSpectrumX) == 0:
		pass

def gaussianWindow(t ,tau, standardDev):
	return np.exp(-(t-tau)**6/(2*standardDev**6))

def cutWithGaussian( initSpectrumX ,initSpectrumY, spike, sigma):
	Ydata = initSpectrumY * gaussianWindow(initSpectrumX, tau = spike, standardDev=sigma) ## ebbe kéne az X tengely is ám!!!!!
	# Ydata = initSpectrumY * scipy.signal.windows.gaussian(len(initSpectrumY), std=sigma)
	return Ydata

# #rossz egyelőre_______________________

# def IFFT(initSpectrumX, initSpectrumY, temporalXdata):
# 	if len(initSpectrumX)>0 and len(initSpectrumY)>0:
# 		Ydata = initSpectrumY
# 		Xdata = temporalXdata

# 		# freqs = np.linspace(1/(deltaT*N),1/N,deltaT)
# 		# freqs = np.fft.rfftfreq(N, deltaT)
# 		yf = np.fft.ifft(Ydata)
# 		angles = np.angle(yf)
# 		return Xdata ,angles

#_____________________________

# a, b = np.loadtxt('examples/fft.txt', unpack = True, delimiter = ',')
# # samX, samY = np.loadtxt('examples/sample.txt', unpack= True, delimiter = ',')
# # refX, refY = np.loadtxt('examples/reference.txt', unpack=True, delimiter = ',')

# xx, yy = FFT(a,b)
# xxx, yyy = IFFT(xx, yy)

# angles = np.angle(yyy)
# plt.plot(a, angles)
# # print(angles)
# plt.show()




