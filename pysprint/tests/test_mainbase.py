''' Testing the main window base functionality. '''

import sys
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from pysprint.logic import MainProgram


class Test(unittest.TestCase):
    def setUp(self):
        pass
        # self.app = QApplication(sys.argv)
        # self.my_interface = MainProgram()
        # self.my_interface.showMaximized()

    def tearDown(self):
        pass
        # self.my_interface.close()

    # def test_swap(self):
    #     x, y = np.loadtxt('test_peak.txt', unpack=True, delimiter=',')
    #     self.my_interface.a = x
    #     self.my_interface.b = y
    #     self.my_interface.redraw_graph()
    #     self.my_interface.fill_table()
    #     swap = self.my_interface.swapButton
    #     QTest.mouseClick(swap, Qt.LeftButton)
    #     np.testing.assert_array_equal(x, self.my_interface.b)
    #     np.testing.assert_array_equal(y, self.my_interface.a)

    # def test_redrawing(self):
    # 	with self.assertRaises(ValueError):
    # 		self.my_interface.a = np.array([1,3,4])
    # 		self.my_interface.b = np.array([])
    # 		self.my_interface.redraw_graph()


if __name__ == "__main__":
    unittest.main()
