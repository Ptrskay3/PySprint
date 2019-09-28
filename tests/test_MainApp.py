import sys
import unittest
try:
	from PyQt5.QtTest import QTest
except:
	raise ImportError('PyQt5 package is missing.')
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

sys.path.append('..')
# sys.modules['lmfit']=None
# import lmfit
from core import evaluate

class TestMainApp(unittest.TestCase):

	def test_SPP(self):
		a, b = 0, 0 
		self.assertEqual(a,b)

if __name__ == '__main__':
	unittest.main()
		
