''' Testing the main window base functionality. '''

import sys
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from pysprint.logic import ImportPage


class TestGen(unittest.TestCase):
    def setUp(self):
        self.app = QApplication(sys.argv)
        self.gen = ImportPage()
        self.gen.showMaximized()

    def tearDown(self):
        self.gen.close()

    def test_gen(self):
        assert len(self.gen.x) == 0
        assert len(self.gen.y) == 0
        assert len(self.gen.ref) == 0
        assert len(self.gen.sam) == 0

    def test_gen1(self):
        self.gen.update_table()
        self.gen.commit()




if __name__ == "__main__":
    unittest.main()
