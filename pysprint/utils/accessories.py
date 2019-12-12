from functools import wraps

import numpy as np
from scipy.interpolate import interp1d

__all__ = ['scipy_disp', 'lmfit_disp', 'findNearest', 'find_closest',
           '_handle_input', 'print_disp', 'fourier_interpolate',
           'between', 'get_closest', 'run_from_ipython']

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def get_closest(xValue, xArray, yArray):
	idx = (np.abs(xArray - xValue)).argmin()
	value = xArray[idx]
	return value, yArray[idx], idx

def between(val, except_around):
	if except_around is None:
		return False
	elif len(except_around) != 2:
		raise ValueError(f'Invalid interval. Try [start, end] instead of {except_around}')
	else:
		lower = float(min(except_around))
		upper = float(max(except_around))
	if val <= upper and val >= lower:
		return True
	return False

def scipy_disp(r):
	for idx in range(len(r)):
		dispersion[idx] = dispersion[idx] * factorial(idx+1)
		dispersion_std[idx] = dispersion_std[idx] * factorial(idx+1)
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
		Ydata = (initSpectrumY - referenceArmY - sampleArmY) / (2 * np.sqrt(referenceArmY * sampleArmY))
	elif (len(referenceArmY) == 0) or (len(sampleArmY) == 0):
		Ydata = initSpectrumY
	elif len(initSpectrumX) == 0:
		raise ValueError('Please load the spectrum!\n')
	elif len(initSpectrumY) == 0:
		raise ValueError('Please load the spectrum!\n')
	else:
		raise TypeError('Input types are wrong.\n')
	return initSpectrumX,  Ydata


def find_closest(xValue, xArray, yArray):
	idx = (np.abs(xArray - xValue)).argmin()
	value = xArray[idx]
	return value, yArray[idx]


def print_disp(f):
    @wraps(f)
    def wrapping(*args, **kwargs):
        disp, disp_std, stri = f(*args, **kwargs)
        labels = ('GD', 'GDD','TOD', 'FOD', 'QOD')  
        for i, (label, disp_item, disp_std_item) in enumerate(zip(labels, disp, disp_std)):
             print(f'{label} = {disp_item} ± {disp_std_item} fs^{i+1}')
        return disp, disp_std, stri
    return wrapping


def fourier_interpolate(x, y):
    ''' Simple linear interpolation for FFTs'''
    xs = np.linspace(x[0], x[-1], len(x))
    intp = interp1d(x, y, kind='linear', fill_value='extrapolate')
    ys = intp(xs)
    return xs, ys
