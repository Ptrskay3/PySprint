''' Testing the main window base functionality. '''

import sys
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from pysprint.logic import GeneratorWindow


class TestGen(unittest.TestCase):
	pass
    # def setUp(self):
    #     self.app = QApplication(sys.argv)
    #     self.gen = GeneratorWindow()
    #     self.gen.showMaximized()

    # def tearDown(self):
    #     self.gen.close()

    # def test_gen(self):
    #     g = self.gen.pushButton_4
    #     self.gen.startLine.setText('4')
    #     self.gen.stopLine.setText('3')
    #     QTest.mouseClick(g, Qt.LeftButton)
    #     assert len(self.gen.xAxisData) == 0
    #     assert len(self.gen.yAxisData) == 0

    # def test_gen1(self):
    #     g = self.gen.pushButton_4
    #     QTest.mouseClick(g, Qt.LeftButton)
    #     assert len(self.gen.xAxisData) != 0
    #     assert len(self.gen.yAxisData) != 0
    #     self.gen.generate_data()


if __name__ == "__main__":
    unittest.main()
