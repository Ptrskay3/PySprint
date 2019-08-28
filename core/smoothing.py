"""
Methods for manipulating the loaded data

"""


import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, gaussian, convolve
from scipy.interpolate import interp1d



def savgolFilter(initSpectrumX, initSpectrumY ,referenceArmY, sampleArmY, window = 101, order = 3):
	if len(initSpectrumX) > 0 and len(referenceArmY)>0 and len(sampleArmY)>0:
		try:
			Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
		except ValueError as error:
			print(error)
		Xdata = initSpectrumX
		xint, yint = interpolateData(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY)
	elif len(initSpectrumY) == 0:
		pass
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = initSpectrumY
		Xdata = initSpectrumX
		xint, yint = interpolateData(initSpectrumX, initSpectrumY, [], [])
	if window > order:
		try:
			if window % 2 == 1:
				fil = savgol_filter(yint, window_length = window, polyorder = order)
				return xint, fil
			else:
				fil = savgol_filter(yint, window_length = window + 1, polyorder = order)
				return xint, fil
		except Exception as e:
			print(e)
	else:
		pass



#too many duplicates, needs a rewrite
def findPeaks(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, proMax = 1, proMin =1, threshold= 0.1):   
	if len(initSpectrumX) > 0 and len(referenceArmY)>0 and len(sampleArmY)>0:
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
		Xdata = initSpectrumX
		maxIndexes = find_peaks(Ydata, prominence = proMax) #a reciprok nem túl elegáns megoldás..
		Ydata = 1/Ydata
		minIndexes = find_peaks(Ydata, prominence = proMin)
		Ydata = 1/Ydata
		test = np.array([]) 
		testX = np.array([])
		for element in Ydata[minIndexes[0]]:
			if element > threshold or element < -threshold:
				test = np.append(test, element)
				ind = np.where(Ydata[minIndexes[0]] == element)
				testX = np.append(testX, Xdata[minIndexes[0]][ind])

		if len(Xdata[maxIndexes[0]]) != len(Ydata[maxIndexes[0]]) or len(testX) != len(test):
			raise ValueError('Something went wrong, try to cut the edges of data.')
		# plt.plot(Xdata, Ydata)
		# plt.plot(Xdata[maxIndexes[0]],Ydata[maxIndexes[0]], 'ro')
		# plt.plot(testX, test, 'b*')
		# plt.show()
		return Xdata[maxIndexes[0]], Ydata[maxIndexes[0]], testX, test
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = initSpectrumY
		Xdata = initSpectrumX
		maxIndexes = find_peaks(Ydata, prominence = proMax)
		Ydata = 1/Ydata 
		minIndexes = find_peaks(Ydata, prominence = proMin)
		Ydata = 1/Ydata
		test = np.array([]) # átdolgozandó még..
		testX = np.array([])
		for element in Ydata[minIndexes[0]]:
			if element > threshold or element < -threshold:
				test = np.append(test, element)
				ind = np.where(Ydata[minIndexes[0]] == element)
				testX = np.append(testX, Xdata[minIndexes[0]][ind])
		if len(Xdata[maxIndexes[0]]) != len(Ydata[maxIndexes[0]]) or len(testX) != len(test):
			raise ValueError('Something went wrong, try to cut the edges of data.')
		# if len(Xdata[maxIndexes[0]]) > Ydata[maxIndexes[0]]:
		# 	XEdited = (Xdata[maxIndexes[0]])[:len(Xdata[maxIndexes[0]])]
		# elif len(Ydata[maxIndexes[0]]) > Xdata[maxIndexes[0]]:
		# 	YEdited = (Xdata[maxIndexes[0]])[:len(Xdata[maxIndexes[0]])]

		# if len(testX)>len(test):
		# 	testXEdit = testX[:len(test)]
		# elif len(test)>len(testX):
		# 	testEdit = test[:len(testX)]
		# plt.plot(Xdata, Ydata)
		# plt.plot(Xdata[maxIndexes[0]],Ydata[maxIndexes[0]], 'ro')
		# plt.plot(testX, test, 'b*')
		# plt.show()
		return Xdata[maxIndexes[0]], Ydata[maxIndexes[0]], testX, test


# b, a, c, d = np.loadtxt('examples/autodetect.txt', unpack= True, delimiter=',', skiprows = 10)
# findPeaks(a,b,c,d)

def convolution(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, standev = 200):
	if len(initSpectrumX) > 0 and len(referenceArmY)>0 and len(sampleArmY)>0:
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
		Xdata = initSpectrumX
		xint, yint = interpolateData(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY)
	elif len(initSpectrumY) == 0:
		pass
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = initSpectrumY
		Xdata = initSpectrumX
		xint, yint = interpolateData(initSpectrumX, initSpectrumY, [], [])
	window = gaussian(len(xint), std=standev)
	smoothed = convolve(yint, window/window.sum(), mode='same')
	return xint, smoothed



def interpolateData(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY):
	if len(initSpectrumX) > 0 and len(referenceArmY)>0 and len(sampleArmY)>0:
		Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
		Xdata = initSpectrumX
		xint = np.linspace(Xdata[0], Xdata[-1], len(Xdata))
		intp = interp1d(Xdata,Ydata, kind='linear')
		yint = intp(xint)
		return xint, yint
	elif len(initSpectrumX) == 0:
		pass
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = initSpectrumY
		Xdata = initSpectrumX
		xint = np.linspace(Xdata[0], Xdata[-1], len(Xdata))
		intp = interp1d(Xdata,Ydata, kind='linear')
		yint = intp(xint)
		return xint, yint


def cutData(initSpectrumX, initSpectrumY, referenceArmY, sampleArmY, startValue=-9999, endValue=9999):
	from .evaluate import findNearest
	if len(initSpectrumX) > 0 and len(referenceArmY)>0 and len(sampleArmY)>0:
			Ydata = (initSpectrumY-referenceArmY-sampleArmY)/(2*np.sqrt(referenceArmY*sampleArmY))
			Xdata = initSpectrumX
	elif len(initSpectrumY) == 0:
		pass
	elif len(referenceArmY) == 0 or len(sampleArmY) == 0:
		Ydata = initSpectrumY
		Xdata = initSpectrumX

	if startValue < endValue:
		lowItem, lowIndex = findNearest(Xdata, startValue)
		highItem, highIndex = findNearest(Xdata, endValue)
		neededIndex = np.where((Xdata>=lowItem) & (Xdata<=highItem))
		return Xdata[neededIndex], Ydata[neededIndex]
	else:
		pass

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    dist = np.abs(array[idx]-value)
    return array[idx], idx

def find_closest(xValue, xArray, yArray):
    value,index = find_nearest(xArray, xValue)
    return value, yArray[index]
# arr = np.array([5,3,0,1,4])
# arrY = np.array([4,5,6,7,8])
# # indee = np.where((arr > 0) & (arr < 4) )

# # print(indee)
# xxx, yyy = cutData(arr, arrY, [], [], startValue = 2, endValue = 5)
# print(xxx, yyy)