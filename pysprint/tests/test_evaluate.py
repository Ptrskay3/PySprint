'''
FIXME: Obviously we need better tests than that..
'''
import sys
import unittest
import numpy as np
import scipy
# import matplotlib.pyplot as plt

from pysprint.core import evaluate
from pysprint.core.dataedits import find_peak
from pysprint import Generator, FFTMethod


class TestEvaluate(unittest.TestCase):

	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_min_max_bases(self):
		a = np.arange(100)
		b = np.arange(100)
		mins = [10,30,50,70,90]
		maxs = [20,40,60,80,100]
		disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx=maxs, minx=mins, fit_order=1, show_graph=False)
		np.testing.assert_almost_equal(disp[0], 0)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method([], b, [], [], 0, maxx=maxs, minx=mins, fit_order=1, show_graph=False)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method([], [], [], [], 0, maxx=maxs, minx=mins, fit_order=1, show_graph=False)
		# with self.assertRaises(ValueError):
			# disp, disp_s, fit = evaluate.min_max_method(a, [], [], [], ref_point = 0, maxx=maxs, minx=mins, fit_order=1, show_graph=False)
		with self.assertRaises(ValueError):
			disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], 0, maxx=maxs, minx=mins, fit_order=6, show_graph=False)
		# with self.assertRaises(np.core._exceptions.UFuncTypeError):
			# disp, disp_s, fit = evaluate.min_max_method(a, b, [], [], ref_point = 0, maxx='a', minx=mins, fit_order=1, show_graph=False)
		# with self.assertRaises(ValueError):
			# disp, disp_s, fit = evaluate.min_max_method(a, b, b, b, ref_point=-50000, maxx=maxs, minx=mins, fit_order=1, show_graph=False)

	def test_min_max_advanced(self):
		# emp = np.zeros(5)
		# j,k = evaluate.min_max_method(np.arange(100), np.arange(100), [], [], ref_point = 10, fit_order = 1, show_graph=False)
		# np.testing.assert_array_equal(emp, j)
		# np.testing.assert_array_equal(emp, k)
		a,b,c,d = np.loadtxt('test_arms.txt', delimiter = ',', unpack = True)
		maxs, _, mins, _ = find_peak(a,b,c,d, proMax=1, proMin=1, threshold=0.4)
		d1, d_s1, fit1 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fit_order = 1, maxx = maxs, minx = mins, show_graph = False)
		np.testing.assert_array_equal(d1[1:], [0,0,0,0])
		np.testing.assert_array_equal(d_s1[1:], [0,0,0,0])
		assert len(d1) == len(d_s1) == 5
		d2, d_s2, fit2 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fit_order = 2, maxx = maxs, minx = mins, show_graph = False)
		np.testing.assert_array_equal(d2[2:], [0,0,0])
		np.testing.assert_array_equal(d_s2[2:], [0,0,0])
		assert len(d2) == len(d_s2) == 5
		d3, d_s3, fit3 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fit_order = 3, maxx = maxs, minx = mins, show_graph = False)
		np.testing.assert_array_equal(d3[3:], [0,0])
		np.testing.assert_array_equal(d_s3[3:], [0,0])
		assert len(d3) == len(d_s3) == 5
		d4, d_s4, fit4 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fit_order = 4, maxx = maxs, minx = mins, show_graph = False)
		np.testing.assert_array_equal(d4[4:], [0])
		np.testing.assert_array_equal(d_s4[4:], [0])
		assert len(d4) == len(d_s4) == 5
		d5, d_s5, fit5 = evaluate.min_max_method(a,b,c,d, ref_point = 2.5, fit_order = 5, maxx = maxs, minx = mins, show_graph = False)
		# np.testing.assert_array_equal(d3[3:], [0,0])
		# np.testing.assert_array_equal(d_s3[3:], [0,0])
		# plt.close('all')
		assert len(d5) == len(d_s5) == 5


			
	def test_cff(self):
		a = np.arange(100)
		b = np.arange(100)
		with self.assertRaises(TypeError):
			evaluate.cff_method(a, b, [], [], ref_point=0 , p0=[1, 1, 1, 1, 1, 1, 1, 1, 1])

	def test_ffts_primitive(self):
		#adapted from scipy's unittests
	    np.random.seed(1534)
	    x = np.random.randn(10) + 1j * np.random.randn(10)
	    fr, yf = evaluate.ifft_method(x, x, interpolate = False)
	    _, y = evaluate.fft_method(yf, yf)
	    np.testing.assert_allclose(y, x)


	def test_ffts_advanced2(self):
		g = Generator(2,2.8,2.4, delay=1500, GDD=2000, pulse_width=25, resolution=0.01)
		g.generate_freq()
		a,b = g.data
		f = FFTMethod(a, b)
		f.ifft()
		f.window(1500, 2920, plot=False)
		f.apply_window()
		f.fft()
		d, _, _ = f.calculate(order = 2, reference_point = 2.4)
		np.testing.assert_array_almost_equal(d, [-1500.01, -1999.79, 0, 0, 0], decimal=2)

	def test_ffts_advanced1(self):
		g = Generator(2,2.8,2.4, delay = 1500, GD = 200, pulse_width = 25, resolution = 0.01)
		g.generate_freq()
		a,b = g.data
		f = FFTMethod(a, b)
		f.ifft()
		f.window(1700, 3300, plot=False)
		f.apply_window()
		f.fft()
		d, _, _ = f.calculate(order = 1, reference_point = 2.4)
		np.testing.assert_array_almost_equal(d, [-1699.99, 0, 0, 0, 0], decimal=2)

	def test_ffts_advanced3(self):
		g = Generator(2,2.8,2.4, delay = 1500, TOD = 40000, pulse_width = 25, resolution = 0.01)
		g.generate_freq()
		a,b = g.data
		f = FFTMethod(a, b)
		f.ifft()
		f.window(2500, 4830, window_order = 12, plot=False)
		f.apply_window()
		f.fft()
		d, _, _ = f.calculate(order = 3, reference_point = 2.4)
		np.testing.assert_array_almost_equal(d, [-1500.03, -0.03, -39996.60, 0, 0], decimal=2)


	def test_ffts_advanced4(self):
		g = Generator(2,2.8,2.4, delay=1500, GDD=2000, FOD=-100000, pulse_width=25, resolution=0.01)
		g.generate_freq()
		a,b = g.data
		f = FFTMethod(a, b)
		f.ifft()
		f.window(1500, 1490, window_order = 8, plot=False)
		f.apply_window()
		f.fft()
		d, _, _ = f.calculate(order = 4, reference_point = 2.4)
		np.testing.assert_array_almost_equal(d, [-1500.00, -1999.95, 0.21, 99995.00, 0], decimal=1)

	def test_ffts_advanced5(self):
		g = Generator(2,2.8,2.4, delay = 1500, QOD = 900000, pulse_width = 25, resolution = 0.01)
		g.generate_freq()
		a,b = g.data
		f = FFTMethod(a, b)
		f.ifft()
		f.window(1600, 2950, window_order = 12, plot=False)
		f.apply_window()
		f.fft()
		d, _, _ = f.calculate(order = 5, reference_point = 2.4)
		np.testing.assert_array_almost_equal(d, [-1499.96, 0.14, -7.88, -15.99, -898920.79], decimal=1)

	def test_windowing(self):
		a,b = np.loadtxt('test_window.txt', unpack = True, delimiter = ',')
		y_data = evaluate.cut_gaussian(a, b, 2.5, 0.2, 6)
		assert len(b) == len(y_data)
		np.testing.assert_almost_equal(y_data[0], 0)
		np.testing.assert_almost_equal(y_data[-1], 0)
		np.testing.assert_almost_equal(np.median(y_data), np.median(b), decimal=2)

	def test_spp(self):
		pass

if __name__ == '__main__':
	unittest.main()