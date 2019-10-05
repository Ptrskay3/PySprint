import sys
import unittest
import numpy as np

sys.path.append('..')

from core.edit_features import *

class TestEdit(unittest.TestCase):

	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_savgol(self):
		pass
	
	def test_peak(self):
		x, y = np.loadtxt('test_peak.txt', delimiter = ',', unpack = True)
		a,b,c,d = find_peak(x, y, [], [], threshold = 0.01, proMin = 0.5, proMax = 0.5)
		assert len(a) == len(b)
		assert len(c) == len(d)
		for val in b:
			assert abs(val) > 0.01
		for val in d:
			assert abs(val) > 0.01

	def test_interpolate(self):
		pass

	def test_cut(self):
		pass

	def test_convolution(self):
		pass

if __name__ == '__main__':
	unittest.main()
		
