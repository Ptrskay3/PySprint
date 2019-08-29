"""
The main logic behind the UI functions.
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QTreeWidget, QTreeWidgetItem, QAbstractItemView,
QDialog, QPushButton, QVBoxLayout, QComboBox, QCheckBox, QLabel,QAction, qApp, QTextEdit, QSpacerItem, QSizePolicy,QHBoxLayout, QGroupBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QIcon, QCursor

from ui.ui import Ui_Interferometry
from ui.generatorUI import Ui_GeneratorWindow
from ui.aboutUI import Help
from ui.mplwidget import MplWidget
from ui.SPPUI import Ui_SPP

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib

from core.evaluate import minMaxMethod, PMCFFMethod, FFT, cutWithGaussian, gaussianWindow , IFFT, argsAndCompute, SPP
from core.smoothing import savgolFilter, findPeaks, convolution, interpolateData, cutData, find_closest
from core.loading import readData
from core.generator import generatorFreq, generatorWave

class MainProgram(QtWidgets.QMainWindow, Ui_Interferometry):
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
        self.calculate.clicked.connect(self.getit)
        self.btn_load.clicked.connect(lambda i: self.loadData(i, self.a))
        self.swapButton.clicked.connect(self.swapAxes)
        self.temporalApplyButton.clicked.connect(self.temporalApply)
        self.commitChanges.clicked.connect(self.commitToData)
        self.resetButton.clicked.connect(self.resetAll)
        self.refreshGraph.clicked.connect(self.redrawGraph)
        self.iReferenceArm.clicked.connect(lambda i: self.referenceArmClicked(i, self.refX))
        self.iSampleArm.clicked.connect(lambda i: self.sampleArmClicked(i, self.samX))
        self.iReferenceArm_2.clicked.connect(lambda i: self.referenceArmClicked(i, self.refX))
        self.iSampleArm_2.clicked.connect(lambda i: self.sampleArmClicked(i, self.samX))
        self.doFFT.clicked.connect(self.fftHandler)
        self.doCut.clicked.connect(self.gaussianCutFunction)
        self.doIFFT.clicked.connect(self.ifftHandler)
        self.actionAbout.triggered.connect(self.openHelp)
        self.actionSave_current_data.triggered.connect(self.saveLoadedData)
        self.actionSave_log_file.triggered.connect(self.saveOutput)
        self.actionExit.triggered.connect(self.close)
        self.actionGenerator.triggered.connect(self.openGenerator)
        self.pushButton.clicked.connect(self.openSPPanel)
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

    def openHelp(self):
        self.window1 = helpWindow(self)
        self.window1.show()

    def openGenerator(self):
        self.window2 = generatorWindow(self)
        self.window2.show()

    def openSPPanel(self):
        self.window3 = SPPWindow(self)
        self.window3.show()

    def messageOutput(self, text):
        self.logOutput.insertPlainText('\n' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ':')
        self.logOutput.insertPlainText('\n {}\n\n'.format(str(text)))
        self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())


    def waitingEffects(function):
        def new_function(self):
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            try:
                return function(self)
            finally:
                QApplication.restoreOverrideCursor()
        return new_function

    def gaussianCutFunction(self):
        if self.gaussianCut.text() == '':
            self.gaussianCut.setText('100')
        if self.gaussianCut2.text() == '':
            self.gaussianCut2.setText('40')
        if len(self.a)>0 and len(self.b)>0:
            xx = cutWithGaussian(self.a ,self.b, spike= float(self.gaussianCut.text()), sigma = float(self.gaussianCut2.text()))
            self.b = xx
            self.redrawGraph()

    def fftHandler(self):
        if len(self.a)>0 and len(self.b)>0:
            self.fftContainer = self.a
            self.a, self.b = FFT(self.a, self.b)
            self.redrawGraph()
            self.messageOutput('FFT applied to data. Some functions may behave differently. The absolute value is plotted.')
        else:
            self.messageOutput('No data is loaded.')

    def ifftHandler(self):
        if len(self.a)>0 and len(self.b)>0 and len(self.fftContainer)>0:
            self.b = IFFT(self.b)
            self.a = self.fftContainer
            self.fftContainer = np.array([])
            self.redrawGraph()
            self.messageOutput('IFFT done. ')
            
    @waitingEffects
    def swapAxes(self):
        self.tableWidget.setRowCount(0)
        if len(self.a)>0:
            self.temp = self.a
            self.a = self.b
            self.b = self.temp
            self.redrawGraph()
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

    def commitToData(self):
        if self.editTab.currentIndex() == 1:
            if self.peaksMax.text() == '':
                self.peaksMax.setText('0.1')
            if self.peaksMin.text() == '':
                self.peaksMin.setText('0.1')
            if self.peaksThreshold.text() == '':
                self.peaksThreshold.setText('0.1')
            try:
                if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                    j, k, l, m = findPeaks(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.maxx = j
                    self.minx = l 
                elif len(self.a) == 0:
                    pass
                elif len(self.refY) == 0 or len(self.samY) == 0:
                    j, k, l, m = findPeaks(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.maxx = j
                    self.minx = l 
                self.messageOutput('Points were recorded for min-max method.')
            except Exception as e:
                self.messageOutput(e)


        if self.editTab.currentIndex() == 0:
            if self.savgolWindow.text() == '':
                self.savgolWindow.setText('51')
            if self.savgolOrder.text() == '':
                self.savgolOrder.setText('3')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                    self.a, self.b = savgolFilter(self.a, self.b ,self.refY, self.samY, window = int(self.savgolWindow.text()), 
                        order = int(self.savgolOrder.text()))
                    self.refY = []
                    self.samY = []
                    self.messageOutput('Reference and sample arm is now merged and the spectrum is normalized.')
                else:
                    self.messageOutput('Data shapes are different. Operation canceled.')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.a, self.b = savgolFilter(self.a, self.b ,[], [], window = int(self.savgolWindow.text()), 
                    order = int(self.savgolOrder.text()))

            self.redrawGraph()

        if self.editTab.currentIndex() == 2:
            if self.convolutionStd.text() == '':
                self.convolutionStd.setText('5')
            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                    self.a, self.b = convolution(self.a, self.b, self.refY, self.samY, standev = float(self.convolutionStd.text()))
                    self.refY = []
                    self.samY = []
                    self.messageOutput('Reference and sample arm is now merged and the spectrum is normalized.')
                else:
                    self.messageOutput('Data shapes are different. Operation canceled.')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.a, self.b = convolution(self.a, self.b ,[], [], standev = float(self.convolutionStd.text()))

            self.redrawGraph()
        
        if self.editTab.currentIndex() == 3:
            if self.sliceStart.text() =='':
                self.sliceStart.setText('-9999')
            if self.sliceStop.text() == '':
                self.sliceStop.setText('9999')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                if len(self.a) == len(self.refY) and len(self.a) == len(self.b):
                    self.a, self.b = cutData(self.a, self.b, self.refY, self.samY, startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                    self.refY = []
                    self.samY = []
                    self.messageOutput('Reference and sample arm is now merged and the spectrum is normalized.')
                else:
                    self.messageOutput('Data shapes are different. Operation canceled')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                try:
                    self.a, self.b  = cutData(self.a, self.b ,[], [], startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                except:
                    self.messageOutput('Invalid values encountered..')
            self.redrawGraph()

    def resetAll(self):
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
        self.messageOutput('Data cleared.')
        self.tableWidget.clear()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Angular frequency", "Intensity"])

    def testt(self):
        print(self.editTab.currentIndex())
        print(self.editTab.currentWidget())

    def temporalApply(self):
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
                    j, k, l, m = findPeaks(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
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
                    j, k, l, m = findPeaks(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                     proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                    self.MplWidget.canvas.axes.plot(self.a, self.b)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.plot(j, k, 'ro')
                    self.MplWidget.canvas.axes.plot(l, m, 'ko')
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
            except Exception as e:
                self.messageOutput(e)


        if self.editTab.currentIndex() == 0:

            if self.savgolWindow.text() == '':
                self.savgolWindow.setText('51')
            if self.savgolOrder.text() == '':
                self.savgolOrder.setText('3')

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                self.MplWidget.canvas.axes.clear()
                try:
                    if len(self.a) == len(self.refY) and len(self.a) == len(self.samY):
                        m, n = savgolFilter(self.a, self.b ,self.refY, self.samY, window = int(self.savgolWindow.text()), 
                            order = int(self.savgolOrder.text()))
                        self.MplWidget.canvas.axes.plot(m, n)
                        self.MplWidget.canvas.axes.grid()
                        self.MplWidget.canvas.axes.set_ylabel("Intensity")
                        # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                        self.MplWidget.canvas.draw()
                    else:
                        self.messageOutput('Data shapes are different. Operation canceled.')


                except:
                    self.messageOutput('Polynomial order must be less than window..')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                self.MplWidget.canvas.axes.clear()
                try:
                    m, n = savgolFilter(self.a, self.b ,[], [], window = int(self.savgolWindow.text()), 
                        order = int(self.savgolOrder.text()))
                    self.MplWidget.canvas.axes.plot(m, n)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                except:
                    self.messageOutput('Polynomial order must be less than window.')

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
                    self.messageOutput('Data shapes are different. Operation canceled.')

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
                    t, w = cutData(self.a, self.b, self.refY, self.samY, startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                    self.MplWidget.canvas.axes.plot(t, w)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                else:
                    self.messageOutput('Data shapes are different. Operation canceled.')
            elif len(self.a) == 0:
                pass
            elif len(self.refY) == 0 or len(self.samY) == 0:
                try:
                    self.MplWidget.canvas.axes.clear()
                    t,w  = cutData(self.a, self.b ,[], [], startValue = float(self.sliceStart.text()),
                     endValue = float(self.sliceStop.text()))
                    self.MplWidget.canvas.axes.plot(t, w)
                    self.MplWidget.canvas.axes.grid()
                    self.MplWidget.canvas.axes.set_ylabel("Intensity")
                    # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                    self.MplWidget.canvas.draw()
                except:
                    self.messageOutput('Invalid values encountered..')

    def redrawGraph(self):
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
                self.messageOutput('Data shapes are different. Operation canceled.')

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
        # print(self.a[1])
        # print(self.b[1])
        # print(self.samY[1])
        # print(self.refY[1])

    @pyqtSlot(float)
    def referenceArmClicked(self, refX, refY):
        options = QFileDialog.Options()
        referenceName, _ = QFileDialog.getOpenFileName(None,"Reference arm spectrum", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if referenceName:
                self.refX, self.refY= np.loadtxt(referenceName, usecols=(0,1), unpack = True, delimiter =',')
        except:
            self.messageOutput('Failed')
    
    @pyqtSlot(float)   
    def sampleArmClicked(self, samX, samY):
        options = QFileDialog.Options()       
        sampleName, _ = QFileDialog.getOpenFileName(None,"Sample arm spectrum", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if sampleName:
                self.samX, self.samY= np.loadtxt(sampleName, usecols=(0,1), unpack = True, delimiter = ',')
          
        except:
            self.messageOutput('Failed')

    @pyqtSlot(float) 
    def loadData(self, a, b): 
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getOpenFileName(None,"Load interferogram", "","All Files (*);;Text Files (*.txt)", options=options)
            try:
                if fileName:
                    self.tableWidget.setRowCount(0)
                    try:
                        self.a, self.b, self.refY, self.samY = readData(fileName)
                    except:
                        self.messageOutput('Auto-detect failed, attempting to load again..')  
                        self.a, self.b = np.loadtxt(fileName, usecols=(0,1), unpack = True, delimiter =',')  
                        self.messageOutput('Done')
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
            
                self.redrawGraph()
                # print(self.refY[:3])
            except Exception as e:
                self.messageOutput(e)
            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()
   
    @waitingEffects
    def getit(self):
        if self.methodWidget.currentIndex() == 2:
            if self.mmPoly.text() == '':
                self.mmPoly.setText('5')
            try:
                disp, disp_std, fit_report = minMaxMethod(self.a, self.b,  self.refY, self.samY, float(self.getSPP.text()), self.maxx, self.minx,
                    int(self.mmPoly.text()), showGraph = self.checkGraph.isChecked())
                labels = ['GD', 'GDD', 'TOD', 'FOD', 'QOD']
                self.messageOutput('Using Min-max method.')
                for item in range(len(disp)):
                    self.logOutput.insertPlainText(' '+ labels[item] +' =  ' + str(disp[item]) +' +/- ' + str(disp_std[item]) + ' 1/fs^'+str(item+1)+'\n')
                if self.printCheck.isChecked():
                    self.messageOutput(fit_report)
                self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())
            except Exception as e:
                self.messageOutput(str(e))
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
                cFF = PMCFFMethod(self.a, self.b ,self.refY, self.samY, 
                    p0=[1,1,1, float(self.initGD.text()), float(self.initGDD.text()), float(self.initTOD.text()), float(self.initFOD.text()),
                    float(self.initQOD.text())]) 
                labels = ['GD', 'GDD', 'TOD', 'FOD', 'QOD']
                self.messageOutput('Using Cosine function fit method..')
                try:
                    for item in range(len(cFF)):
                        self.logOutput.insertPlainText(' '+ labels[item] +' =  ' + str(cFF[item]) +'  1/fs^'+str(item+1)+'\n')
                    self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())
                except Exception as e:
                    self.messageOutput('You might need to provide initial guess for parameters.')
                    self.messageOutput(e)
            except Exception as e:
                self.messageOutput(e)

        if self.methodWidget.currentIndex() == 0 or self.methodWidget.currentIndex() == 3:
            self.messageOutput('not implemented')
            
    def saveOutput(self):
        options = QFileDialog.Options()
        name = QFileDialog.getSaveFileName(self, 'Save File','','Text(*.txt)', options=options)
        try:
            with open(name[0], 'w') as f:
                text = self.logOutput.toPlainText()
                f.write(text)
        except:
            pass

    def saveLoadedData(self):
        options = QFileDialog.Options()
        name = QFileDialog.getSaveFileName(self, 'Save File','','Text(*.txt)', options=options)
        try:
            with open(name[0], 'w') as f:
                if len(self.a)>0 and len(self.refY)>0 and len(self.samY)>0:
                    np.savetxt(name[0], np.transpose([self.a, self.b, self.refY, self.samY]), delimiter=',')
                elif len(self.refY) == 0 or len(self.samY == 0):
                    np.savetxt(name[0], np.transpose([self.a, self.b]), delimiter = ',')
                else:
                    self.messageOutput('Something went wrong.')
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
    #     self.redrawGraph()
    #     self.tutorial1 = QMessageBox.about(self, "Tutorial", "I loaded in some example data for you. You can manipulate the data on the right panel and use the methods below. ")



class helpWindow(QtWidgets.QMainWindow, Help):
    def __init__(self, parent=None):
        super(helpWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.exbtn.clicked.connect(self.close)


class generatorWindow(QtWidgets.QMainWindow, Ui_GeneratorWindow):
    xAxisData = np.array([])
    yAxisData = np.array([])
    refData = np.array([])
    samData = np.array([])

    def __init__(self, parent=None):
        super(generatorWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.closeButton.clicked.connect(self.close)
        # self.pushButton_3.clicked.connect(self.emitData)
        self.pushButton_4.clicked.connect(self.generateData)
        self.pushButton_2.clicked.connect(self.saveAs)
        self.armCheck.setChecked(True)
        self.delimiterLine.setText(',')

    def previewData(self):
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

    def generateData(self):
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
        
        self.previewData()
        
    def saveAs(self):
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
    def __init__(self, parent=None):
        super(SPPWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.treeWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.loadButton.clicked.connect(self.loadUp)
        self.treeWidget.itemSelectionChanged.connect(self.fillSPP)
        self.pushButton.clicked.connect(self.recordDelay)
        self.treeWidget.itemSelectionChanged.connect(self.previewData)
        self.pushButton_7.clicked.connect(self.deleteItem)
        self.pushButton_2.clicked.connect(self.pressed)
        self.pushButton_3.clicked.connect(self.released)
        self.pushButton_4.clicked.connect(self.editSPP)
        self.pushButton_6.clicked.connect(self.doSPP)
        self.pushButton_5.clicked.connect(self.cleanUp)

    def doSPP(self):
        self.widget.canvas.axes.clear()
        if self.fitOrderLine.text() == '':
            self.fitOrderLine.setText('4')
        try:
            xs, ys, disp, disp_std, bestFit = SPP(delays = self.delays, omegas = self.xpoints, fitOrder = int(self.fitOrderLine.text()))
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
            self.messageOutput('Some values might be missing. Fit order must be lower or equal than the number of data points.\n' + str(e))

    def onclick(self, event):
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
            self.messageOutput('Clicks are no longer recorded.')
        return 
    
    def messageOutput(self, text):
        self.messageBox.insertPlainText(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ':')
        self.messageBox.insertPlainText('\n {}\n\n'.format(str(text)))
        self.messageBox.verticalScrollBar().setValue(self.messageBox.verticalScrollBar().maximum())

    def pressed(self):
        self.cid = self.widget.canvas.mpl_connect('button_press_event', self.onclick)
        self.pushButton_2.setText('Activated')

    def released(self):
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
        self.previewData()

    def fillSPP(self):
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

    def editSPP(self):
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
        self.previewData()

    def previewData(self):
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

    def loadUp(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None,"Load interferogram", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            actualCount = self.treeWidget.topLevelItemCount()
            if fileName:
                xx, yy, vv, ww = readData(fileName)
                self.xData.append(xx)
                self.yData.append(yy)
                self.yRef.append(vv)
                self.ySam.append(ww)
                l1 = QTreeWidgetItem([fileName.split('/')[-1]])
                self.treeWidget.addTopLevelItem(l1)
            self.previewData()
        except Exception as e:
            print(e)

    def deleteItem(self):
        try:
            curr = self.treeWidget.currentIndex().row()
            #ez nem biztos hogy kell
            self.delays[curr] = 0
            self.xpoints[curr] = 0
            self.ypoints[curr] = 0
            self.treeWidget.currentItem().setHidden(True)
        except:
            pass

    def cleanUp(self):
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

    def recordDelay(self):
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
