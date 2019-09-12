"""
The main logic behind the UI functions.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QDialogButtonBox, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QTreeWidget, QTreeWidgetItem, QAbstractItemView,
QDialog, QPushButton, QVBoxLayout, QComboBox, QCheckBox, QLabel,QAction, qApp, QTextEdit, QSpacerItem, QSizePolicy,QHBoxLayout, QGroupBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, pyqtSlot, QSettings
from PyQt5.QtGui import QIcon, QCursor

from ui.ui import Ui_Interferometry
from ui.generatorUI import Ui_GeneratorWindow
from ui.aboutUI import Help
from ui.mplwidget import MplWidget
from ui.SPPUI import Ui_SPP
from ui.settings_dialog import Ui_SettingsWindow

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib

from core.evaluate import min_max_method, cff_method, fft_method, cut_gaussian, gaussian_window , ifft_method, spp_method
from core.edit_features import savgol, find_peak, convolution, interpolate_data, cut_data, find_closest
from core.loading import read_data
from core.generator import generatorFreq, generatorWave

class MainProgram(QtWidgets.QMainWindow, Ui_Interferometry):
    """ The main window class, opened when main.py is run."""
    samX = np.array([])
    samY = np.array([])
    refX = np.array([])
    refY = np.array([])
    a = np.array([])
    b = np.array([])
    temp = np.array([])
    fftContainer = np.array([])
    minx = np.array([])
    maxx = np.array([])

    def __init__(self, parent=None):
        super(MainProgram, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.settings = QSettings("_settings.ini", QSettings.IniFormat)
        self.settings.setFallbacksEnabled(False)
        self.calculate.clicked.connect(self.get_it)
        self.btn_load.clicked.connect(lambda i: self.load_data(i, self.a))
        self.swapButton.clicked.connect(self.swap_axes)
        self.temporalApplyButton.clicked.connect(self.apply_on_plot)
        self.commitChanges.clicked.connect(self.commit_to_data)
        self.resetButton.clicked.connect(self.reset_all)
        self.refreshGraph.clicked.connect(self.redraw_graph)
        self.iReferenceArm.clicked.connect(lambda i: self.ref_arm_clicked(i, self.refX))
        self.iSampleArm.clicked.connect(lambda i: self.sam_arm_clicked(i, self.samX))
        self.iReferenceArm_2.clicked.connect(lambda i: self.ref_arm_clicked(i, self.refX))
        self.iSampleArm_2.clicked.connect(lambda i: self.sam_arm_clicked(i, self.samX))
        self.doFFT.clicked.connect(self.fft_handler)
        self.doCut.clicked.connect(self.gauss_cut_func)
        self.doIFFT.clicked.connect(self.ifft_handler)
        self.actionAbout.triggered.connect(self.open_help)
        self.actionSave_current_data.triggered.connect(self.save_curr_data)
        self.actionSave_log_file.triggered.connect(self.save_output)
        self.actionExit.triggered.connect(self.close)
        self.actionGenerator.triggered.connect(self.open_generator)
        self.actionSettings.triggered.connect(self.open_settings)
        self.pushButton.clicked.connect(self.open_sppanel)

        self.show_dialog()
        
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(5)
        self.tableWidget.setHorizontalHeaderLabels(["Angular frequency", "Intensity"])
        self.tableWidget.setSizeAdjustPolicy(
        QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.savgolTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.peakTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.convolTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.cutTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.sppTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.cffTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.mmTab.setStyleSheet(" background-color: rgb(240,240,240);")
        self.fftTab.setStyleSheet(" background-color: rgb(240,240,240);")
        # self.actionUnit_converter.triggered.connect(self.runTutorial)
        self.btn_load.setToolTip('Load in data. Can be different type (see documentation)')
        self.swapButton.setToolTip('Swaps the two columns and redraws graph.')
        self.resetButton.setToolTip('Erases all data from memory.')
        self.refreshGraph.setToolTip('Redraws graph with the current data. If arms are loaded shows normalized graph.')
        self.temporalApplyButton.setToolTip('Shows preview of data with the given parameters.')
        self.commitChanges.setToolTip('Applies changes to data.')
        self.doFFT.setToolTip('Perfoms FFT on current data.')
        self.doCut.setToolTip('Applies a gaussian window with parameters defined on the right.')
        self.doIFFT.setToolTip('Perfoms IFFT on current data.')
        self.checkGraph.setToolTip('Show a plot with the dataset and fitted curve.')
        self.mmPoly.setToolTip('Assumed maximum order of dispersion.')
        self.printCheck.setToolTip('Include lmfit report in the log.')

    # FIXME: IS IT POSSIBLE TO EXEC THE MAINWINDOW BEFORE SHOWING THIS MESSAGE?

    def show_dialog(self):
        self.cb = QCheckBox('Do not show this message again.', self.centralwidget)
        self.msgbox = QMessageBox(self)
        self.msgbox.setText('Welcome to Interferometry!\nDo not forget to set the defaults at Edit --> Settings. For more details, see documentation.')
        self.msgbox.setWindowTitle('Interferometry')
        self.msgbox.setCheckBox(self.cb)
        self.msgbox.setStandardButtons(QMessageBox.Ok)
        if self.settings.value('show') == 'True':
            self.msgbox.exec_()
        if self.cb.isChecked():
            self.settings.setValue('show', False)



    def open_help(self):
        """ Opens up help window."""
        self.window1 = HelpWindow(self)
        self.window1.show()
        print(self.settings.value('GD'))

    def open_generator(self):
        """ Opens up generator window"""
        self.window2 = GeneratorWindow(self)
        self.window2.show()

    def open_sppanel(self):
        """ Opens up SPP Interface"""
        self.window3 = SPPWindow(self)
        self.window3.show()

    def open_settings(self):
        self.window4 = SettingsWindow(self)
        self.window4.show()

    def msg_output(self, text):
        """ Prints to the log dialog"""
        self.logOutput.insertPlainText('\n' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ':')
        self.logOutput.insertPlainText('\n {}\n\n'.format(str(text)))
        self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())


    def waiting_effects(function):
        """ Decorator to show loading cursor"""
        def new_function(self):
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            try:
                return function(self)
            finally:
                QApplication.restoreOverrideCursor()
        return new_function

    def gauss_cut_func(self):
        """ On FFT tab perfoms a cut with 6 order gaussian """
        if self.gaussianCut.text() == '':
            self.gaussianCut.setText('100')
        if self.gaussianCut2.text() == '':
            self.gaussianCut2.setText('40')
        if len(self.a)>0 and len(self.b)>0:
            xx = cut_gaussian(self.a ,self.b, spike= float(self.gaussianCut.text()), sigma = float(self.gaussianCut2.text()))
            self.b = xx
            self.redraw_graph()

    def fft_handler(self):
        """ On FFT tab perfoms FFT on currently loaded data"""
        if len(self.a)>0 and len(self.b)>0:
            self.fftContainer = self.a
            self.a, self.b = fft_method(self.a, self.b)
            self.redraw_graph()
            self.msg_output('FFT applied to data. Some functions may behave differently. The absolute value is plotted.')
        else:
            self.msg_output('No data is loaded.')

    def ifft_handler(self):
        """ On FFt tab perfoms IFFT on currently loaded data""" 
        if len(self.a)>0 and len(self.b)>0 and len(self.fftContainer)>0:
            self.b = ifft_method(self.b)
            self.a = self.fftContainer
            self.fftContainer = np.array([])
            self.redraw_graph()
            self.msg_output('IFFT done. ')
            
    @waiting_effects
    def swap_axes(self):
        """ Changes the x and y axis"""
        self.tableWidget.setRowCount(0)
        if len(self.a)>0:
            self.temp = self.a
            self.a = self.b
            self.b = self.temp
            self.redraw_graph()
            if len(self.a)<400:
                        for row_number in range(len(self.a)):
                            self.tableWidget.insertRow(row_number)
                            for item in range(len(self.a)):
                                self.tableWidget.setItem(item, 0, QtWidgets.QTableWidgetItem(str(self.a[item])))
                            for item in range(len(self.b)):
                                self.tableWidget.setItem(item, 1, QtWidgets.QTableWidgetItem(str(self.b[item])))
            else:
                for row_number in range(400):
                    self.tableWidget.insertRow(row_number)
                    for item in range(400):
                        self.tableWidget.setItem(item, 0, QtWidgets.QTableWidgetItem(str(self.a[item])))
                    for item in range(400):
                        self.tableWidget.setItem(item, 1, QtWidgets.QTableWidgetItem(str(self.b[item])))
            self.tableWidget.resizeRowsToContents()
            self.tableWidget.resizeColumnsToContents()

    def commit_to_data(self):
        """ On the data manipulation tab applies the current function with the given parameters to the loaded dataset."""
        if self.editTab.currentIndex() == 1:
            if self.peaksMax.text() == '':
                self.peaksMax.setText('0.1')
            if self.peaksMin.text() == '':
                self.peaksMin.setText('0.1')
            if self.peaksThreshold.text() == '':
                self.peaksThreshold.setText('0.1')
            try:
                if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                    j, k, l, m = find_peak(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.maxx = j
                    self.minx = l 
                elif len(self.a) == 0:
                    pass
                elif len(self.refY) == 0 or len(self.samY) == 0:
                    j, k, l, m = find_peak(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.maxx = j
                    self.minx = l 
                self.msg_output('Points were recorded for min-max method.')
            except Exception as e:
                self.msg_output(e)


        if self.editTab.currentIndex() == 0:
            if self.savgolWindow.text() == '':
                self.savgolWindow.setText('51')
            if self.savgolOrder.text() == '':
                self.savgolOrder.setText('3')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                    self.a, self.b = savgol(self.a, self.b ,self.refY, self.samY, window = int(self.savgolWindow.text()), 
                        order = int(self.savgolOrder.text()))
                    self.refY = []
                    self.samY = []
                    self.msg_output('Reference and sample arm is now merged and the spectrum is normalized.')
                else:
                    self.msg_output('Data shapes are different. Operation canceled.')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.a, self.b = savgol(self.a, self.b ,[], [], window = int(self.savgolWindow.text()), 
                    order = int(self.savgolOrder.text()))

            self.redraw_graph()

        if self.editTab.currentIndex() == 2:
            if self.convolutionStd.text() == '':
                self.convolutionStd.setText('5')
            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                    self.a, self.b = convolution(self.a, self.b, self.refY, self.samY, standev = float(self.convolutionStd.text()))
                    self.refY = []
                    self.samY = []
                    self.msg_output('Reference and sample arm is now merged and the spectrum is normalized.')
                else:
                    self.msg_output('Data shapes are different. Operation canceled.')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.a, self.b = convolution(self.a, self.b ,[], [], standev = float(self.convolutionStd.text()))

            self.redraw_graph()
        
        if self.editTab.currentIndex() == 3:
            if self.sliceStart.text() =='':
                self.sliceStart.setText('-9999')
            if self.sliceStop.text() == '':
                self.sliceStop.setText('9999')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.b):
                    self.a, self.b = cut_data(self.a, self.b, self.refY, self.samY, startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                    self.refY = []
                    self.samY = []
                    self.msg_output('Reference and sample arm is now merged and the spectrum is normalized.')
                else:
                    self.msg_output('Data shapes are different. Operation canceled')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                try:
                    self.a, self.b  = cut_data(self.a, self.b ,[], [], startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                except:
                    self.msg_output('Invalid values encountered..')
            self.redraw_graph()

    def reset_all(self):
        """ Clears all the loaded data and plot."""
        self.a = []
        self.b = []
        self.refY = []
        self.samY = []
        self.fftContainer = []
        self.minx = []
        self.maxx = []
        self.temp = []
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.draw()
        self.msg_output('Data cleared.')
        self.tableWidget.clear()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Angular frequency", "Intensity"])

    def testt(self):
        print(self.editTab.currentIndex())
        print(self.editTab.currentWidget())

    def apply_on_plot(self):
        """ On the data manipulation tab applies the current function but only shows the plot and doesn't commit the changes."""
        if self.editTab.currentIndex() == 1:
            if self.peaksMax.text() == '':
                self.peaksMax.setText('0.1')
            if self.peaksMin.text() == '':
                self.peaksMin.setText('0.1')
            if self.peaksThreshold.text() == '':
                self.peaksThreshold.setText('0.1')
            try:
                if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                    self.MplWidget.canvas.axes.clear()
                    j, k, l, m = find_peak(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.plot(self.a, ((self.b-self.refY-self.samY)/(2*np.sqrt(self.refY*self.samY))))
                    self.MplWidget.canvas.axes.plot(j, k, 'ro')
                    self.MplWidget.canvas.axes.plot(l, m, 'ko')
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                elif len(self.a) == 0:
                    pass
                elif len(self.refY) == 0 or len(self.samY) == 0:
                    self.MplWidget.canvas.axes.clear()
                    j, k, l, m = find_peak(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.MplWidget.canvas.axes.plot(self.a, self.b)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.plot(j, k, 'ro')
                    self.MplWidget.canvas.axes.plot(l, m, 'ko')
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
            except Exception as e:
                self.msg_output(e)


        if self.editTab.currentIndex() == 0:

            if self.savgolWindow.text() == '':
                self.savgolWindow.setText('51')
            if self.savgolOrder.text() == '':
                self.savgolOrder.setText('3')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                self.MplWidget.canvas.axes.clear()
                try:
                    if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                        m, n = savgol(self.a, self.b ,self.refY, self.samY, window = int(self.savgolWindow.text()), 
                            order = int(self.savgolOrder.text()))
                        self.MplWidget.canvas.axes.plot(m, n)
                        self.MplWidget.canvas.axes.grid()
                        self.MplWidget.canvas.axes.set_ylabel("Intensity")
                        # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                        self.MplWidget.canvas.draw()
                    else:
                        self.msg_output('Data shapes are different. Operation canceled.')


                except:
                    self.msg_output('Polynomial order must be less than window..')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.MplWidget.canvas.axes.clear()
                try:
                    m, n = savgol(self.a, self.b ,[], [], window = int(self.savgolWindow.text()), 
                        order = int(self.savgolOrder.text()))
                    self.MplWidget.canvas.axes.plot(m, n)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                except:
                    self.msg_output('Polynomial order must be less than window.')

        if self.editTab.currentIndex() == 2:
            if self.convolutionStd.text() == '':
                self.convolutionStd.setText('5')
            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                self.MplWidget.canvas.axes.clear()
                if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                    u, v = convolution(self.a, self.b, self.refY, self.samY, standev = float(self.convolutionStd.text()))
                    self.MplWidget.canvas.axes.plot(u, v)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                else:
                    self.msg_output('Data shapes are different. Operation canceled.')

            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.MplWidget.canvas.axes.clear()
                u, v = convolution(self.a, self.b ,[], [], standev = float(self.convolutionStd.text()))
                self.MplWidget.canvas.axes.plot(u, v)
                self.MplWidget.canvas.axes.grid()
                self.MplWidget.canvas.axes.set_ylabel("Intensity")
                # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                self.MplWidget.canvas.draw()

        if self.editTab.currentIndex() == 3:
            if self.sliceStart.text() =='':
                self.sliceStart.setText('-9999')
            if self.sliceStop.text() == '':
                self.sliceStop.setText('9999')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                    self.MplWidget.canvas.axes.clear()
                    t, w = cut_data(self.a, self.b, self.refY, self.samY, startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                    self.MplWidget.canvas.axes.plot(t, w)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                else:
                    self.msg_output('Data shapes are different. Operation canceled.')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                try:
                    self.MplWidget.canvas.axes.clear()
                    t,w  = cut_data(self.a, self.b ,[], [], startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                    self.MplWidget.canvas.axes.plot(t, w)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                except:
                    self.msg_output('Invalid values encountered..')

    def redraw_graph(self):
        """ Function to update the plot"""
        if (len(self.a) > 0) and (len(self.refY) > 0) and (len(self.samY) > 0) and (len(self.b)>0):
            if len(self.a) == len(self.samY) and len(self.a) == len(self.refY):
                Ydata = (self.b-self.refY-self.samY)/(2*np.sqrt(self.refY*self.samY))
                Xdata = self.a
                self.MplWidget.canvas.axes.clear()
                if np.iscomplexobj(Ydata):
                    self.MplWidget.canvas.axes.plot(Xdata, np.abs(Ydata))
                else:
                    self.MplWidget.canvas.axes.plot(Xdata, Ydata)
                self.MplWidget.canvas.axes.set_ylabel("Intensity")
                # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                self.MplWidget.canvas.axes.grid()
                self.MplWidget.canvas.draw()
            else:
                self.msg_output('Data shapes are different. Operation canceled.')

        elif len(self.a) == 0:
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.text(0.42, 0.47, 'No data to plot')
            self.MplWidget.canvas.axes.grid()
            self.MplWidget.canvas.draw()
        elif len(self.refY) == 0 or len(self.samY) == 0:
            Ydata = self.b
            Xdata = self.a
            self.MplWidget.canvas.axes.clear()
            if np.iscomplexobj(Ydata):
                self.MplWidget.canvas.axes.plot(Xdata, np.abs(Ydata))
            else:
                self.MplWidget.canvas.axes.plot(Xdata, Ydata)
            # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
            self.MplWidget.canvas.axes.grid()
            self.MplWidget.canvas.draw()


    @pyqtSlot(float)
    def ref_arm_clicked(self, refX, refY):
        """ Loads in the reference arm data"""
        options = QFileDialog.Options()
        referenceName, _ = QFileDialog.getOpenFileName(None,"Reference arm spectrum", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if referenceName:
                self.refX, self.refY= np.loadtxt(referenceName, usecols=(0,1), unpack = True, delimiter =',')
        except:
            self.msg_output('Failed')
    
    @pyqtSlot(float)   
    def sam_arm_clicked(self, samX, samY):
        """ Loads in the sample arm data"""
        options = QFileDialog.Options()       
        sampleName, _ = QFileDialog.getOpenFileName(None,"Sample arm spectrum", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if sampleName:
                self.samX, self.samY= np.loadtxt(sampleName, usecols=(0,1), unpack = True, delimiter = ',')
          
        except:
            self.msg_output('Failed')

    @pyqtSlot(float) 
    def load_data(self, a, b): 
        """ Loads in the data with AI. If that fails, loads in manually with np.loadtxt."""
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None,"Load interferogram", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if fileName:
                self.tableWidget.setRowCount(0)
                try:
                    self.a, self.b, self.refY, self.samY = read_data(fileName)
                except:
                    self.msg_output('Auto-detect failed, attempting to load again..')  
                    self.a, self.b = np.loadtxt(fileName, usecols=(0,1), unpack = True, delimiter =',')  
                    self.msg_output('Done')
                if len(self.a)<100:
                    for row_number in range(len(self.a)):
                        self.tableWidget.insertRow(row_number)
                        for item in range(len(self.a)):
                            self.tableWidget.setItem(item, 0, QtWidgets.QTableWidgetItem(str(self.a[item])))
                        for item in range(len(self.b)):
                            self.tableWidget.setItem(item, 1, QtWidgets.QTableWidgetItem(str(self.b[item])))
                else:
                    for row_number in range(100):
                        self.tableWidget.insertRow(row_number)
                        for item in range(100):
                            self.tableWidget.setItem(item, 0, QtWidgets.QTableWidgetItem(str(self.a[item])))
                        for item in range(100):
                            self.tableWidget.setItem(item, 1, QtWidgets.QTableWidgetItem(str(self.b[item])))
        
            self.redraw_graph()
            # print(self.refY[:3])
        except Exception as e:
            self.msg_output(e)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
   
    @waiting_effects
    def get_it(self):
        """ If everything's set, calculates the dispersion."""
        if self.methodWidget.currentIndex() == 2:
            if self.mmPoly.text() == '':
                self.mmPoly.setText('5')
            try:
                disp, disp_std, fit_report = min_max_method(self.a, self.b,  self.refY, self.samY, float(self.getSPP.text()), self.maxx, self.minx,
                    int(self.mmPoly.text()), showGraph = self.checkGraph.isChecked())
                labels = ['GD', 'GDD', 'TOD', 'FOD', 'QOD']
                self.msg_output('Using Min-max method.')
                for item in range(len(disp)):
                    self.logOutput.insertPlainText(' '+ labels[item] +' =  ' + str(disp[item]) +' +/- ' + str(disp_std[item]) + ' 1/fs^'+str(item+1)+'\n')
                if self.printCheck.isChecked():
                    self.msg_output(fit_report)
                self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())
            except Exception as e:
                self.msg_output(str(e))
        if self.methodWidget.currentIndex() == 1:
            if self.initGD.text() == '':
                self.initGD.setText('1')
            if self.initGDD.text() == '':
                self.initGDD.setText('1')
            if self.initTOD.text() == '':
                self.initTOD.setText('1')
            if self.initFOD.text() == '':
                self.initFOD.setText('1')
            if self.initQOD.text() == '':
                self.initQOD.setText('1')
            try:
                cFF = cff_method(self.a, self.b ,self.refY, self.samY, 
                    p0=[1,1,1, float(self.initGD.text()), float(self.initGDD.text()), float(self.initTOD.text()), float(self.initFOD.text()),
                    float(self.initQOD.text())]) 
                labels = ['GD', 'GDD', 'TOD', 'FOD', 'QOD']
                self.msg_output('Using Cosine function fit method..')
                try:
                    for item in range(len(cFF)):
                        self.logOutput.insertPlainText(' '+ labels[item] +' =  ' + str(cFF[item]) +'  1/fs^'+str(item+1)+'\n')
                    self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())
                except Exception as e:
                    self.msg_output('You might need to provide initial guess for parameters.')
                    self.msg_output(e)
            except Exception as e:
                self.msg_output(e)

        if self.methodWidget.currentIndex() == 3:
            self.msg_output('not implemented')
        if  self.methodWidget.currentIndex() == 0:
            self.msg_output('Please use the interface for SPP method.')
            
    def save_output(self):
        """ Saves the logoutput to a txt file."""
        options = QFileDialog.Options()
        name = QFileDialog.getSaveFileName(self, 'Save File','','Text(*.txt)', options=options)
        try:
            with open(name[0], 'w') as f:
                text = self.logOutput.toPlainText()
                f.write(text)
        except:
            pass

    def save_curr_data(self):
        """ Saves the currently loaded data into a .txt file."""
        options = QFileDialog.Options()
        name = QFileDialog.getSaveFileName(self, 'Save File','','Text(*.txt)', options=options)
        try:
            with open(name[0], 'w') as f:
                if len(self.a)>0 and len(self.refY)>0 and len(self.samY)>0:
                    np.savetxt(name[0], np.transpose([self.a, self.b, self.refY, self.samY]), delimiter=',')
                elif len(self.refY) == 0 or len(self.samY == 0):
                    np.savetxt(name[0], np.transpose([self.a, self.b]), delimiter = ',')
                else:
                    self.msg_output('Something went wrong.')
        except:
            pass


    # def runTutorial(self):
    #     self.a, self.b, self.refY, self.samY = readData('examples/fft.txt')
    #     if len(self.a)<400:
    #         for row_number in range(len(self.a)):
    #             self.tableWidget.insertRow(row_number)
    #             for item in range(len(self.a)):
    #                 self.tableWidget.setItem(item, 0, QtWidgets.QTableWidgetItem(str(self.a[item])))
    #             for item in range(len(self.b)):
    #                 self.tableWidget.setItem(item, 1, QtWidgets.QTableWidgetItem(str(self.b[item])))
    #     else:
    #         for row_number in range(400):
    #             self.tableWidget.insertRow(row_number)
    #             for item in range(400):
    #                 self.tableWidget.setItem(item, 0, QtWidgets.QTableWidgetItem(str(self.a[item])))
    #             for item in range(400):
    #                 self.tableWidget.setItem(item, 1, QtWidgets.QTableWidgetItem(str(self.b[item])))
    #     self.tableWidget.resizeRowsToContents()
    #     self.tableWidget.resizeColumnsToContents()
    #     self.redraw_graph()
    #     self.tutorial1 = QMessageBox.about(self, "Tutorial", "I loaded in some example data for you. You can manipulate the data on the right panel and use the methods below. ")



class HelpWindow(QtWidgets.QMainWindow, Help):
    """ Class for the help window."""
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.exbtn.clicked.connect(self.close)


class GeneratorWindow(QtWidgets.QMainWindow, Ui_GeneratorWindow):
    """ Class for the generator window."""
    xAxisData = np.array([])
    yAxisData = np.array([])
    refData = np.array([])
    samData = np.array([])

    def __init__(self, parent=None):
        super(GeneratorWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.closeButton.clicked.connect(self.close)
        # self.pushButton_3.clicked.connect(self.emitData)
        self.pushButton_4.clicked.connect(self.generate_data)
        self.pushButton_2.clicked.connect(self.save_as)
        self.armCheck.setChecked(True)
        self.delimiterLine.setText(',')

    def preview_data(self):
        """Function to update plot."""
        if (len(self.xAxisData) > 0) and (len(self.refData) > 0) and (len(self.samData) > 0) and (len(self.yAxisData)>0):
            if len(self.xAxisData) == len(self.samData) and len(self.xAxisData) == len(self.refData):
                Ydata = (self.yAxisData-self.refData-self.samData)/(2*np.sqrt(self.refData*self.samData))
                Xdata = self.xAxisData
                self.plotWidget.canvas.axes.clear()
                self.plotWidget.canvas.axes.plot(Xdata, Ydata, 'r')
                self.plotWidget.canvas.axes.grid()
                self.plotWidget.canvas.draw()
            else:
                pass
        elif len(self.xAxisData) == 0:
            self.plotWidget.canvas.axes.clear()
            self.plotWidget.canvas.axes.text(0.42, 0.47, 'No data to plot')
            self.plotWidget.canvas.axes.grid()
            self.plotWidget.canvas.draw()
        elif len(self.refData) == 0 or len(self.samData) == 0:
            Ydata = self.yAxisData
            Xdata = self.xAxisData
            self.plotWidget.canvas.axes.clear()
            self.plotWidget.canvas.axes.plot(Xdata, Ydata, 'r')
            self.plotWidget.canvas.axes.grid()
            self.plotWidget.canvas.draw()

    def generate_data(self):
        """ Function to generate the dataset. If fails, the button changes to red."""
        if self.startLine.text()=='':
            self.startLine.setText('2')
        if self.stopLine.text()=='':
            self.stopLine.setText('3')
        if self.centerLine.text()=='':
            self.centerLine.setText('2.5')
        if self.pulseLine.text()=='':
            self.pulseLine.setText('0.002')
        if self.resolutionLine.text()=='':
            self.resolutionLine.setText('0.1')
        if self.delayLine.text()=='':
            self.delayLine.setText('0')
        if self.GDLine.text()=='':
            self.GDLine.setText('0')
        if self.GDDLine.text()=='':
            self.GDDLine.setText('200')
        if self.TODLine.text()=='':
            self.TODLine.setText('4000')
        if self.FODLine.text()=='':
            self.FODLine.setText('0')
        if self.QODLine.text()=='':
            self.QODLine.setText('0')
        if self.delimiterLine.text == '':
            self.delimiterLine.setText(',')

        if self.comboBox.currentText() == 'frequency':
            try:
                self.pushButton_4.setStyleSheet('background-color: None')
                self.xAxisData, self.yAxisData, self.refData, self.samData =  generatorFreq(start = float(self.startLine.text()),
                    stop = float(self.stopLine.text()), center = float(self.centerLine.text()), delay = float(self.delayLine.text()), 
                    GD = float(self.GDLine.text()), GDD = float(self.startLine.text()), TOD = float(self.TODLine.text()), FOD = float(self.FODLine.text()), 
                    QOD = float(self.QODLine.text()), resolution = float(self.resolutionLine.text()), delimiter = self.delimiterLine.text(), pulseWidth = float(self.pulseLine.text()), 
                    includeArms = self.armCheck.isChecked())
            except:
                self.pushButton_4.setStyleSheet(" background-color: rgb(240,0,0); color: rgb(255,255,255);")

        if self.comboBox.currentText() == 'wavelength':
            try:
                self.pushButton_4.setStyleSheet('background-color: None')
                self.xAxisData, self.yAxisData, self.refData, self.samData =  generatorWave(start = float(self.startLine.text()),
                    stop = float(self.stopLine.text()), center = float(self.centerLine.text()), delay = float(self.delayLine.text()), 
                    GD = float(self.GDLine.text()), GDD = float(self.startLine.text()), TOD = float(self.TODLine.text()), FOD = float(self.FODLine.text()), 
                    QOD = float(self.QODLine.text()), resolution = float(self.resolutionLine.text()), delimiter = self.delimiterLine.text(), pulseWidth = float(self.pulseLine.text()), 
                    includeArms = self.armCheck.isChecked())
            except:
                self.pushButton_4.setStyleSheet(" background-color: rgb(240,0,0); color: rgb(255,255,255);")
        
        self.preview_data()
        
    def save_as(self):
        """ Function to save the generated dataset."""
        if self.delimiterLine.text == '':
            self.delimiterLine.setText(',')
        options = QFileDialog.Options()
        name = QFileDialog.getSaveFileName(self, 'Save File','','Text(*.txt)', options=options)
        # print(len(self.xAxisData))
        try:
            with open(name[0], 'w') as f:
                if self.armCheck.isChecked():
                    # np.savetxt(name[0], np.transpose(self.xAxisData ,self.yAxisData, self.refData, self.samData), 
                    # header = 'freq, int, ref, sam', delimiter = ',', comments ='')
                    np.savetxt(name[0], np.column_stack((self.xAxisData, self.yAxisData, self.refData ,self.samData)), delimiter = str(self.delimiterLine.text())                        )
                    # pd.to_csv(name[0], columns = [self.xAxisData ,self.yAxisData, self.refData, self.samData])
                else:
                   np.savetxt(name[0], np.column_stack((self.xAxisData, self.yAxisData)), delimiter = str(self.delimiterLine.text()))
        except:
            pass

    
class SPPWindow(QtWidgets.QMainWindow, Ui_SPP):
    """ Class for the SPP Interface"""
    def __init__(self, parent=None):
        super(SPPWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.treeWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.loadButton.clicked.connect(self.load_up)
        self.treeWidget.itemSelectionChanged.connect(self.fill_SPP)
        self.pushButton.clicked.connect(self.record_delay)
        self.treeWidget.itemSelectionChanged.connect(self.preview_data)
        self.pushButton_7.clicked.connect(self.delete_item)
        self.pushButton_2.clicked.connect(self.pressed)
        self.pushButton_3.clicked.connect(self.released)
        self.pushButton_4.clicked.connect(self.edit_SPP)
        self.pushButton_6.clicked.connect(self.do_SPP)
        self.pushButton_5.clicked.connect(self.clean_up)

    def do_SPP(self):
        """ Applies the SPP method to the given dataset, if fails it leaves a message."""
        self.widget.canvas.axes.clear()
        if self.fitOrderLine.text() == '':
            self.fitOrderLine.setText('4')
        try:
            xs, ys, disp, disp_std, bestFit = spp_method(delays = self.delays, omegas = self.xpoints, fitOrder = int(self.fitOrderLine.text()))
            self.widget.canvas.axes.clear()
            self.widget.canvas.axes.plot(xs, ys,'o', label = 'data')
            self.widget.canvas.axes.plot(xs, bestFit, 'r--', label = 'fit')
            self.widget.canvas.axes.legend()
            self.widget.canvas.axes.set_ylabel('fs')
            self.widget.canvas.axes.grid()
            self.widget.canvas.draw()
            self.GDSPP.setText(str(disp[0]) + ' +/- ' + str(disp_std[0])+ ' 1/fs')
            self.GDDSPP.setText(str(disp[1]) + ' +/- ' + str(disp_std[1])+ ' 1/fs^2')
            self.TODSPP.setText(str(disp[2]) + ' +/- ' + str(disp_std[2])+ ' 1/fs^3')
            self.FODSPP.setText(str(disp[3]) + ' +/- ' + str(disp_std[3])+ ' 1/fs^4')
            self.QODSPP.setText(str(disp[4]) + ' +/- ' + str(disp_std[4])+ ' 1/fs^5')
        except Exception as e:
            self.msg_output('Some values might be missing. Fit order must be lower or equal than the number of data points.\n' + str(e))

    def on_clicked(self, event):
        """ Function to record clicks on plot."""
        global ix, iy
        ix, iy = event.xdata, event.ydata
        curr = self.treeWidget.currentIndex().row()
        x = self.xData[curr]
        if len(self.ySam[curr]) == 0:
            y = self.yData[curr]
        else:
            y = (self.yData[curr]-self.yRef[curr]-self.ySam[curr])/(2*np.sqrt(self.yRef[curr]*self.ySam[curr]))
        ix, iy= find_closest(ix, x, y)

        self.xtemporal.append(ix)
        self.ytemporal.append(iy)

        colors = ['black','green','blue','yellow']
        self.widget.canvas.axes.scatter(ix, iy, cmap=matplotlib.colors.ListedColormap(colors), s = 80, zorder = 99)
        self.widget.canvas.draw()

        if len(self.xtemporal) == 4:
            self.widget.canvas.mpl_disconnect(self.cid)
            self.msg_output('Clicks are no longer recorded.')
        return 
    
    def msg_output(self, text):
        """ Prints messages to the log widget."""
        self.messageBox.insertPlainText(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ':')
        self.messageBox.insertPlainText('\n {}\n\n'.format(str(text)))
        self.messageBox.verticalScrollBar().setValue(self.messageBox.verticalScrollBar().maximum())

    def pressed(self):
        """ Function to record clicks on plot."""
        self.cid = self.widget.canvas.mpl_connect('button_press_event', self.on_clicked)
        self.pushButton_2.setText('Activated')

    def released(self):
        """ Function to record clicked points on the plot"""
        self.widget.canvas.mpl_disconnect(self.cid)
        curr = self.treeWidget.currentIndex().row()
        while len(self.xtemporal)< 4:
            self.xtemporal.append(None)
        while len(self.ytemporal) < 4:
            self.ytemporal.append(None)
        self.ypoints[curr] = self.ytemporal
        self.xpoints[curr]= self.xtemporal
        self.ytemporal = []
        self.xtemporal = []
        self.pushButton_2.setText('Clickable SPP')
        try:
            self.SPP1.setText(str(self.xpoints[curr][0]))
        except:
            pass
        try:
            self.SPP2.setText(str(self.xpoints[curr][1]))
        except:
            pass
        try:
            self.SPP3.setText(str(self.xpoints[curr][2]))
        except:
            pass
        try:
            self.SPP4.setText(str(self.xpoints[curr][3]))
        except:
            pass
        self.preview_data()

    def fill_SPP(self):
        """ Function to fill up SPP lines. If not given, pass."""
        curr = self.treeWidget.currentIndex().row()
        try:
            self.SPP1.setText(str(self.xpoints[curr][0]))
        except:
            self.SPP1.setText(str(None))
        try:
            self.SPP2.setText(str(self.xpoints[curr][1]))
        except:
            self.SPP2.setText(str(None))
        try:
            self.SPP3.setText(str(self.xpoints[curr][2]))
        except:
            self.SPP3.setText(str(None))
        try:
            self.SPP4.setText(str(self.xpoints[curr][3]))
        except:
            self.SPP4.setText(str(None))

    def edit_SPP(self):
        """ Function to allow user to type in or edit SPP's"""
        curr = self.treeWidget.currentIndex().row()
        try:
            xval1, yval1 = find_closest(float(self.SPP1.text()), self.xData[curr], self.yData[curr])
            self.xtemporal.append(xval1)
            self.ytemporal.append(yval1)
        except:
            pass
        try:
            xval2, yval2 = find_closest(float(self.SPP2.text()), self.xData[curr], self.yData[curr])
            self.xtemporal.append(xval2)
            self.ytemporal.append(yval2)
        except:
            pass
        try:
            xval3, yval3 = find_closest(float(self.SPP3.text()), self.xData[curr], self.yData[curr])
            self.xtemporal.append(xval3)
            self.ytemporal.append(yval3)
        except:
            pass
        try:
            xval4, yval4 = find_closest(float(self.SPP4.text()), self.xData[curr], self.yData[curr])
            self.xtemporal.append(xval4)
            self.ytemporal.append(yval4)
        except:
            pass

        while len(self.xtemporal)< 4:
            self.xtemporal.append(None)
        while len(self.ytemporal) < 4:
            self.ytemporal.append(None)
        self.ypoints[curr] = self.ytemporal
        self.xpoints[curr]= self.xtemporal
        self.ytemporal = []
        self.xtemporal = []
        self.preview_data()

    def preview_data(self):
        """ Function to update plot."""
        curr = self.treeWidget.currentIndex().row()
        self.delayLine.setText('')
        if curr == -1:
            pass
        else:
            try:
                if (len(self.xData[curr]) > 0) and (len(self.yRef[curr]) > 0) and (len(self.ySam[curr]) > 0) and (len(self.yData[curr])>0):
                    if len(self.xData[curr]) == len(self.ySam[curr]) and len(self.xData[curr]) == len(self.yRef[curr]):
                        Ydata = (self.yData[curr]-self.yRef[curr]-self.ySam[curr])/(2*np.sqrt(self.yRef[curr]*self.ySam[curr]))
                        Xdata = self.xData[curr]
                        self.widget.canvas.axes.clear()
                        self.widget.canvas.axes.plot(Xdata, Ydata, 'r')
                        try:
                            if self.xpoints[curr][0] == 0:
                                pass
                            else:
                                colors = ['blue','orange','green','purple']
                                self.widget.canvas.axes.scatter(self.xpoints[curr], self.ypoints[curr], color = colors, s = 80, zorder = 99)
                        except:
                            pass
                        self.widget.canvas.axes.grid()
                        self.widget.canvas.draw()
                    else:
                        pass
                elif len(self.xData[curr]) == 0:
                    self.widget.canvas.axes.clear()
                    self.widget.canvas.axes.text(0.42, 0.47, 'No data to plot')
                    self.widget.canvas.axes.grid()
                    self.widget.canvas.draw()
                elif len(self.yRef[curr]) == 0 or len(self.ySam[curr]) == 0:
                    Ydata = self.yData[curr]
                    Xdata = self.xData[curr]
                    self.widget.canvas.axes.clear()
                    self.widget.canvas.axes.plot(Xdata, Ydata, 'r')
                    try:
                        if self.xpoints[curr][0] == 0:
                            pass
                        else:
                            colors = ['blue','orange','green','purple']
                            self.widget.canvas.axes.scatter(self.xpoints[curr], self.ypoints[curr], color = colors, s = 80, zorder = 99)
                    except:
                        pass
                    self.widget.canvas.axes.grid()
                    self.widget.canvas.draw()
                try:
                    self.delayLine.setText(str(self.delays[curr]))
                except:
                    print('not assigned')
            except:
                pass

    def load_up(self):
        """ Function to load file into Tree widget"""
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None,"Load interferogram", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            actualCount = self.treeWidget.topLevelItemCount()
            if fileName:
                xx, yy, vv, ww = read_data(fileName)
                self.xData.append(xx)
                self.yData.append(yy)
                self.yRef.append(vv)
                self.ySam.append(ww)
                l1 = QTreeWidgetItem([fileName.split('/')[-1]])
                self.treeWidget.addTopLevelItem(l1)
            self.preview_data()
        except Exception as e:
            print(e)

    def delete_item(self):
        """ Function to delete a file from the Tree widget."""
        try:
            curr = self.treeWidget.currentIndex().row()
            #ez nem biztos hogy kell
            self.delays[curr] = 0
            self.xpoints[curr] = 0
            self.ypoints[curr] = 0
            self.treeWidget.currentItem().setHidden(True)
        except:
            pass

    def clean_up(self):
        """ Deletes all the data which is loaded in."""
        self.xData = []
        self.yData = []
        self.ySam = []
        self.yRef = []
        self.xtemporal = []
        self.ytemporal = []
        self.xpoints = [[None]]*20
        self.ypoints = [[None]]*20
        self.delays = np.array([None]*20)
        self.cid = None
        self.treeWidget.clear() 

    def record_delay(self):
        """ Function which allows user to type in delays."""
        curr = self.treeWidget.currentIndex().row()
        if curr == -1:
            pass
        elif curr == 0:
            try:
                self.delays[0]= float(self.delayLine.text())
            except:
                pass
        else:
            try:
                self.delays[curr] = float(self.delayLine.text())
            except:
                pass


class SettingsWindow(QtWidgets.QMainWindow, Ui_SettingsWindow):
    def __init__(self, parent=None):
        super(SettingsWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.pushButton.clicked.connect(self.close)
        self.settings = QSettings("_settings.ini", QSettings.IniFormat)
        self.settings.setFallbacksEnabled(False)
        self.resize(self.settings.value('size', QtCore.QSize(270, 225)))
        self.move(self.settings.value('pos', QtCore.QPoint(50, 50)))
        self.def_GD.setText(self.settings.value('GD'))
        self.def_GDD.setText(self.settings.value('GDD'))
        self.def_TOD.setText(self.settings.value('TOD'))
        self.def_FOD.setText(self.settings.value('FOD'))
        self.def_QOD.setText(self.settings.value('QOD'))
        self.pushButton_2.clicked.connect(self.reset_event)


    def closeEvent(self, e):
        self.settings.setValue('GD', self.def_GD.text())
        self.settings.setValue('GDD', self.def_GDD.text())
        self.settings.setValue('TOD', self.def_TOD.text())
        self.settings.setValue('FOD', self.def_FOD.text())
        self.settings.setValue('QOD', self.def_QOD.text())
        self.settings.setValue('size', self.size())
        self.settings.setValue('pos', self.pos())
        e.accept()

    def reset_event(self):
        self.def_GD.setText('0')
        self.def_GDD.setText('0')
        self.def_TOD.setText('0')
        self.def_FOD.setText('0')
        self.def_QOD.setText('0')
        self.label_7.setVisible(True)


