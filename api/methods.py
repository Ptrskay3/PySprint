"""
This file is the main API to use Interferometry without the UI.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt 

sys.path.append('..')

from core.evaluate import min_max_method, cff_method, fft_method, cut_gaussian, gaussian_window , ifft_method, spp_method, args_comp
from core.edit_features import savgol, find_peak, convolution, interpolate_data, cut_data, find_closest
from core.generator import generatorFreq, generatorWave
from core.cff_fitting import FitOptimizer, cos_fit1, cos_fit2, cos_fit3, cos_fit5, cos_fit4

class Generator(object):

	def __init__(self, start, stop, center, delay=0, GD=0, GDD=0, TOD=0, FOD=0, QOD=0, resolution=0.1, delimiter=',',
		         pulseWidth=10, includeArms=False):
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
		self.pulseWidth = pulseWidth
		self.includeArms = includeArms
		self.x = np.array([])
		self.y = np.array([])
		self.ref = np.array([])
		self.sam = np.array([])

	def generate_freq(self):
		self.x, self.y, self.ref, self.sam = generatorFreq(self.start, self.stop, self.center, self.delay, self.GD,
															    self.GDD, self.TOD, self.FOD, self.QOD,
						   										self.resolution, self.delimiter, self.pulseWidth, self.includeArms)
		if len(self.ref) != 0:
			self.y =  (self.y - self.ref - self.sam)/(2*np.sqrt(self.sam*self.ref))


	def generate_wave(self):
		self.x, self.y, self.ref, self.sam = generatorWave(self.start, self.stop, self.center, self.delay, self.GD,
															    self.GDD, self.TOD, self.FOD, self.QOD,
						   										self.resolution, self.delimiter, self.pulseWidth, self.includeArms)
		if len(self.ref) != 0:
			self.y =  (self.y - self.ref - self.sam)/(2*np.sqrt(self.sam*self.ref))
			
	def show(self):
		plt.figure()
		plt.plot(self.x, self.y, 'r')
		plt.grid()
		plt.show()

	def save(self, name, path=None):
		if path == None:
			np.savetxt('{}.txt'.format(name), np.transpose([self.x, self.y, self.ref, self.sam]))
		else:
			np.savetxt('{}/{}.txt'.format(path, name), np.transpose([self.x, self.y, self.ref, self.sam]))


class Dataset(object):

	def __init__(self, x, y, ref, sam):
		self.x = x
		self.y = y
		self.ref = ref
		self.sam = sam
		if not isinstance(self.x, np.ndarray):
			try:
				self.x = np.asarray(self.x)
			except:
				raise ValueError('Values missing')
		if not isinstance(self.y, np.ndarray):
			try:
				self.y = np.asarray(self.y)
			except:
				raise ValueError('Values missing')
		if not isinstance(self.ref, np.ndarray):
			try:
				self.ref = np.asarray(self.ref)
			except:
				pass
		if not isinstance(self.sam, np.ndarray):
			try:
				self.sam = np.asarray(self.sam)
			except:
				pass
		if len(self.ref) == 0:
			self.y_norm = self.y
		else:
			self.y_norm = (self.y - self.ref - self.sam)/(2*np.sqrt(self.sam*self.ref))


	def show(self):
		plt.figure()
		plt.plot(self.x, self.y_norm)
		plt.grid()
		plt.show()