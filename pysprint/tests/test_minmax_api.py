import sys
import unittest
from unittest.mock import patch

import numpy as np

from pysprint import MinMaxMethod



class TestEvaluate(unittest.TestCase):

	def setUp(self):
		self.x = np.arange(1, 1000, 1)
		self.y = np.sin(self.x)

	def tearDown(self):
		pass

	@patch('matplotlib.pyplot.show')
	def test_edit_session(self, mock_show):
		# ifg = MinMaxMethod(self.x, self.y)
		# ifg.init_edit_session(engine='cwt')
		# ifg.init_edit_session(engine='slope')
		# ifg.init_edit_session(engine='normal')
		# mock_show.assert_called()
		# with self.assertRaises(ValueError):
		# 	ifg.init_edit_session(engine='dfssdfasdf')
		pass

		# this does not work on Azure Pipelines.. should be fixed



if __name__ == '__main__':
	unittest.main()
