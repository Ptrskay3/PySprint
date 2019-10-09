import numpy as np

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