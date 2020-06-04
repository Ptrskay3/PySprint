"""
This should implement the loading test for API.
"""
import sys

sys.path.append("..")

import unittest
import numpy as np

# from pysprint.core import loading


class TestLoading(unittest.TestCase):
    def setUp(self):
        # self.x, self.y, self.v, self.w = np.loadtxt('method_test_w_arms.txt',
        # delimiter = ',', unpack = True)
        pass

    def tearDown(self):
        pass

    # def test_SPP(self):
    # 	a, b = 0, 0
    # 	self.assertEqual(a,b)

    # def test_loading(self):
    # 	self.x, self.y, self.v, self.w = loading.read_data('load_test_1.txt')
    # 	np.testing.assert_array_equal(self.x, np.array([0,0,11]))
    # 	np.testing.assert_array_equal(self.y, np.array([1,1,10]))
    # 	np.testing.assert_array_equal(self.v, np.array([2,2,10]))
    # 	np.testing.assert_array_equal(self.w, np.array([3,3,10]))
    # 	self.x, self.y, self.v, self.w = loading.read_data('load_test_2.txt')
    # 	np.testing.assert_array_equal(self.x, np.array([1,10]))
    # 	np.testing.assert_array_equal(self.y, np.array([0,11]))
    # 	np.testing.assert_array_equal(self.v, np.array([2,10]))
    # 	np.testing.assert_array_equal(self.w, np.array([4,22]))
    # 	with self.assertRaises(FileNotFoundError):
    # 		self.x, self.y, self.v, self.w = loading.read_data('doesnt_exist.txt')


if __name__ == "__main__":
    unittest.main()
