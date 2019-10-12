from functools import wraps

import numpy as np


def scipy_disp(r):
	for idx in range(len(r)):
		dispersion[idx] =  dispersion[idx] / factorial(idx+1) 
		dispersion_std[idx] =  dispersion_std[idx] / factorial(idx+1)
	return dispersion, dispersion_std

def lmfit_disp(r):
	dispersion, dispersion_std = [], []
	for name, par in r:
		dispersion.append(par.value)
		dispersion_std.append(par.stderr)
	return dispersion, dispersion_std

def findNearest(array, value):
	#Finds the nearest element to the given value in the array
	#returns tuple: (element, element's index)
	
    array = np.asarray(array)
    idx = (np.abs(value - array)).argmin()
    return array[idx], idx

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


def find_closest(xValue, xArray, yArray):
	idx = (np.abs(xArray-xValue)).argmin()
	value = xArray[idx]
	return value, yArray[idx]

def print_disp(f):
    @wraps(f)
    def wrapping(*args, **kwargs):
        disp, disp_std, stri = f(*args, **kwargs)
        labels = ['GD', 'GDD','TOD', 'FOD', 'QOD']
        for i in range(len(labels)):
        	print(labels[i] + ' = ' + str(disp[i]) +  ' +/- ' + str(disp_std[i]) + ' 1/fs^{}'.format(i+1))
        return disp, disp_std, stri
    return wrapping