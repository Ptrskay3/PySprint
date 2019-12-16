'''
FIXME: Obviously we need better tests than that..
'''
import sys
import unittest

sys.modules['lmfit'] = None

import numpy as np
import scipy

from pysprint.core import evaluate
from pysprint.core.dataedits import find_peak
from pysprint import Generator, FFTMethod


class TestEvaluateNoLmfit(unittest.TestCase):

	def setUp(self):
		_has_lmfit = False

	def tearDown(self):
		pass

	def test_min_max_bases(self):
		a = np.arange(100)
		b = np.arange(100)
		mins = [10,30,50,70,90]
		maxs = [20,40,60,80,100]
		disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
		np.testing.assert_almost_equal(disp[0], -0.3127272727301358)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method([], b, [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method([], [], [], [], 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
		# with self.assertRaises(ValueError):
			# disp, disp_s, fit = evaluate.min_max_method(a, [], [], [], ref_point = 0, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx=maxs, minx=mins, fitOrder=6, showGraph=False)
		# with self.assertRaises(np.core._exceptions.UFuncTypeError):
			# disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], ref_point = 0, maxx='a', minx=mins, fitOrder=1, showGraph=False)
		# with self.assertRaises(ValueError):
			# disp, disp_s, fit = evaluate.min_max_method(a, b, b, b, ref_point=-50000, maxx=maxs, minx=mins, fitOrder=1, showGraph=False)

	def test_min_max_advanced(self):
		# mp = np.zeros(5)
		# j,k = evaluate.min_max_method(np.arange(100), np.arange(100), [], [], ref_point = 10, fitOrder = 1, showGraph=False)
		# np.testing.assert_array_equal(emp, j)
		# np.testing.assert_array_equal(emp, k)
		a,b,c,d = np.loadtxt('test_arms.txt', delimiter = ',', unpack = True)
		maxs, _, mins, _ = find_peak(a,b,c,d, proMax=1, proMin=1, threshold=0.4)
		d1, d_s1, fit1 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fitOrder = 1, maxx = maxs, minx = mins, showGraph = False)
		np.testing.assert_array_equal(d1[1:], [0,0,0,0])
		np.testing.assert_array_equal(d_s1[1:], [0,0,0,0])
		assert len(d1) == len(d_s1) == 5
		d2, d_s2, fit2 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fitOrder = 2, maxx = maxs, minx = mins, showGraph = False)
		np.testing.assert_array_equal(d2[2:], [0,0,0])
		np.testing.assert_array_equal(d_s2[2:], [0,0,0])
		assert len(d2) == len(d_s2) == 5
		d3, d_s3, fit3 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fitOrder = 3, maxx = maxs, minx = mins, showGraph = False)
		np.testing.assert_array_equal(d3[3:], [0,0])
		np.testing.assert_array_equal(d_s3[3:], [0,0])
		assert len(d3) == len(d_s3) == 5
		d4, d_s4, fit4 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fitOrder = 4, maxx = maxs, minx = mins, showGraph = False)
		np.testing.assert_array_equal(d4[4:], [0])
		np.testing.assert_array_equal(d_s4[4:], [0])
		assert len(d4) == len(d_s4) == 5
		d5, d_s5, fit5 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fitOrder = 5, maxx = maxs, minx = mins, showGraph = False)
		# np.testing.assert_array_equal(d3[3:], [0,0])
		# np.testing.assert_array_equal(d_s3[3:], [0,0])
		# plt.close('all')
		assert len(d5) == len(d_s5) == 5


			
	def test_cff(self):
		_has_lmfit = False
		a = np.arange(100)
		b = np.arange(100)
		with self.assertRaises(TypeError):
			evaluate.cff_method(a, b, [], [], ref_point=0 , p0=[1, 1, 1, 1, 1,1, 1, 1, 1])

	def test_ffts_primitive(self):
		#adapted from scipy's unittests
	    scipy.random.seed(1534)
	    x = scipy.randn(10) + 1j * scipy.randn(10)
	    fr, yf = evaluate.ifft_method(x, x, interpolate = False)
	    _, y = evaluate.fft_method(yf,yf)
	    np.testing.assert_allclose(y, x)

	def test_ffts_advanced(self):
		pass # will be added..  
		# g = Generator(2,2.8,2.4, delay=1500, GDD=2000, pulse_width=25, resolution=0.01)
		# g.generate_freq()
		# a,b = g.unpack()
		# f = FFTMethod(a, b)
		# f.ifft()
		# f.window(1500, 2920)
		# f.apply_window()
		# f.fft()
		# d, _, _ = f.calculate(fit_order = 2, reference_point = 2.4)
		# np.testing.assert_array_equal(d, [-1500.01, -1999.79, 0, 0, 0])

	def test_windowing(self):
		_has_lmfit = False
		a,b = np.loadtxt('test_window.txt', unpack = True, delimiter = ',')
		y_data = evaluate.cut_gaussian(a,b, 2.5, 0.2, 6)
		assert len(b) == len(y_data)
		np.testing.assert_almost_equal(y_data[0], 0)
		np.testing.assert_almost_equal(y_data[-1], 0)
		np.testing.assert_almost_equal(np.median(y_data), np.median(b), decimal = 2)

	def test_spp(self):
		pass

if __name__ == '__main__':
	unittest.main()