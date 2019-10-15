import sys
import unittest
import numpy as np

sys.path.append('..')

from pysprint.core.generator import generatorFreq, generatorWave, C_LIGHT

class TestGenerator(unittest.TestCase):

	def setUp(self):
		assert C_LIGHT == 299.793

	def tearDown(self):
		pass

	def test_errors(self):
		with self.assertRaises(ValueError):
			generatorFreq(start = 1, stop = 2, center = 3, delay = 0)
		with self.assertRaises(ValueError):
			generatorFreq(start = 5, stop = 2, center = 3, delay = 0)
		with self.assertRaises(ValueError):
			generatorFreq(start = 1, stop = -1, center = 3, delay = 0)
		with self.assertRaises(ValueError):
			generatorFreq(start = 1, stop = 3, center = 2, delay = 0, pulseWidth = -20)
		with self.assertRaises(ValueError):
			generatorWave(start = 400, stop = 800, center = 600, delay = 0, resolution = 1000)

	def test_freq(self):
		a,b,c,d = generatorFreq(1,2,1.5, delay = 0, includeArms = True)
		np.testing.assert_array_equal(c,d)
		assert len(a) == len(b)
		e, f, _, _ = generatorFreq(1,2,1.5, delay = 0)
		assert len(e) == len(f)

	def test_wave(self):
		a,b,c,d = generatorWave(1,2,1.5, delay = 0, includeArms = True)
		np.testing.assert_array_equal(c,d)
		assert len(a) == len(b)
		e, f, _, _ = generatorWave(1,2,1.5, delay = 0)
		assert len(e) == len(f)

if __name__ == '__main__':
	unittest.main()
		
