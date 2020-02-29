'''
FIXME: Obviously we need better tests than that..
'''

import unittest
from unittest.mock import patch

import numpy as np

import pysprint
from pysprint.core.evaluate import spp_method
from pysprint import SPPMethod


class TestEvaluate(unittest.TestCase):

	def setUp(self):

		self.delays = np.array([-1700.,-1700.,-1500., -1500., -1300., -1300., -1100., -1100.,  -900.,  -900.,
		  -700.,  -700.,  -500.,  -500.,  -300.,  -300.,  -100.,  -100.,   100.,   100.,
		   300.])

		self.omegas = np.array([2.15686148, 2.55201208, 2.16708089, 2.54264428, 2.17900354, 2.53072164,
		 2.19092619, 2.51965061, 2.20284884, 2.50772796, 2.21732634, 2.49325046,
		 2.23180384, 2.47536649, 2.24968781, 2.45918575, 2.27097825, 2.43874693,
		 2.29993325, 2.40894031, 2.33825604])

	def test_spp_in_core(self):
		x, y, d, ds, _ = spp_method(self.delays, self.omegas, fitOrder=2, from_raw=True, reference_point=2.355)
		np.testing.assert_array_equal(d, [258.84297727172856, -21.572879102888976, -100426.4054547129, 0, 0])
	
	def test_spp_from_raw_api(self):
		with patch('matplotlib.pyplot.show') as p:
			ifgs = SPPMethod('')
			ifgs.set_data(self.delays, self.omegas)
			d, ds, _ = ifgs.calculate(2.355, 2, show_graph=True)
			np.testing.assert_array_equal(d, [258.84297727172856, -21.572879102888976, -100426.4054547129, 0, 0])


if __name__ == '__main__':
	unittest.main()