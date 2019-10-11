'''
FIXME:Obviously we need better tests than that..
'''
import sys

sys.path.append('..')

import unittest
import numpy as np

from pysprint.core import evaluate


class TestLoading(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass

	def test_min_max(self):
		a = np.arange(100)
		b = np.arange(100)
		mins = [10,30,50,70,90]
		maxs = [20,40,60,80,100]
		disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
		np.testing.assert_almost_equal(disp[0], -0.3127272727301358)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method([], b, [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
			disp, disp_s, fit = evaluate.min_max_method([], [], [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
			disp, disp_s, fit = evaluate.min_max_method(a, [], [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
			disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx=maxs, minx=mins, fitOrder=6, showGraph=False)
			disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx='a', minx=mins, fitOrder=1, showGraph=False)
			disp, disp_s, fit = evaluate.min_max_method(a, b, b, b, -50000, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
			
	def test_cff(self):
		a = np.arange(100)
		b = np.arange(100)
		with self.assertRaises(TypeError):
			evaluate.cff_method(a, b, [], [], ref_point=0 , p0=[1, 1, 1, 1, 1,1, 1, 1, 1])

	def test_fft(self):
		pass

	def test_spp(self):
		pass

if __name__ == '__main__':
	unittest.main()