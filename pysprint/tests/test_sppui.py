''' Testing the main window base functionality. '''
import sys
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from pysprint.logic import SPPWindow


class TestSPP(unittest.TestCase):
    def setUp(self):
        self.app = QApplication(sys.argv)
        self.spp = SPPWindow()
        self.spp.show()

    def tearDown(self):
        self.spp.close()

    def test_clean(self):
        self.spp.clean_up()
        self.spp.preview_data()
        self.spp.record_delay()


if __name__ == "__main__":
    unittest.main()
