###################################################
#
#
# 
#
#
##################################################


from ui import Ui_Interferometry
from generatorUI import Ui_GeneratorWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pandas as pd
import time
from mplwidget import MplWidget
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, 
QDialog, QPushButton, QVBoxLayout, QComboBox, QCheckBox, QLabel,QAction, qApp, QTextEdit, QSpacerItem, QSizePolicy,QHBoxLayout, QGroupBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QIcon, QCursor
from datetime import datetime
from evaluate import minMaxMethod, PMCFFMethod, FFT, cutWithGaussian, gaussianWindow , IFFT, argsAndCompute
from help import Help
from smoothing import savgolFilter, findPeaks, convolution, interpolateData, cutData
from loadingData import readData
from generatorCore import generatorFreq, generatorWave

class mainProgram(QtWidgets.QMainWindow, Ui_Interferometry):
    samX = np.array([])
    samY = np.array([])
    refX = np.array([])
    refY = np.array([])
    a = np.array([])
    b = np.array([])
    temp = np.array([])
    fftContainer = np.array([])

    def __init__(self, parent=None):
        super(mainProgram, self).__init__(parent)
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

    def openHelp(self):
        self.window1 = helpWindow(self)
        self.window1.show()

    def openGenerator(self):
        self.window2 = generatorWindow(self)
        self.window2.show()

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
            # self.temp = self.a
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
            ## ez majd az MM methoddal kell összekötni, jelenleg nem csinál semmit


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

            if len(self.a) > 0 and len(self.refY)>0 and len(self.samY)>0:
                self.MplWidget.canvas.axes.clear()
                j, k, l, m = findPeaks(self.a, self.b, self.refY, self.samY, proMax = float(self.peaksMax.text()),
                 proMin = float(self.peaksMin.text()), threshold = float(self.peaksThreshold.text()))
                self.MplWidget.canvas.axes.grid()
                self.MplWidget.canvas.axes.plot(self.a, ((self.b-self.refY-self.samY)/(2*np.sqrt(self.refY*self.samY))))
                self.MplWidget.canvas.axes.plot(j, k, 'ko')
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
                self.MplWidget.canvas.axes.plot(j, k, 'ko')
                self.MplWidget.canvas.axes.plot(l, m, 'ko')
                self.MplWidget.canvas.axes.set_ylabel("Intensity")
                # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
                self.MplWidget.canvas.draw()


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
            try:
                mMm = minMaxMethod(self.a, self.b,  self.refY, self.samY, float(self.getSPP.text()), showGraph = False)
                labels = ['GD', 'GDD', 'TOD', 'FOD', 'QOD']
                self.messageOutput('Using Min-max method.')
                for item in range(len(mMm)-1):
                    self.logOutput.insertPlainText(' '+ labels[item] +' =  ' + str(mMm[item+1]) +'  1/fs^'+str(item+1)+'\n')
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

    def __init__(self, parent = None):
        super(generatorWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.closeButton.clicked.connect(self.close)
        # self.pushButton_4.clicked.connect(self.previewData)
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
                self.plotWidget.canvas.axes.plot(Xdata, Ydata)
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
            self.plotWidget.canvas.axes.plot(Xdata, Ydata)
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
            self.xAxisData, self.yAxisData, self.refData, self.samData =  generatorWave(start = float(self.startLine.text()),
                stop = float(self.stopLine.text()), center = float(self.centerLine.text()), delay = float(self.delayLine.text()), 
                GD = float(self.GDLine.text()), GDD = float(self.startLine.text()), TOD = float(self.TODLine.text()), FOD = float(self.FODLine.text()), 
                QOD = float(self.QODLine.text()), resolution = float(self.resolutionLine.text()), delimiter = self.delimiterLine.text(), pulseWidth = float(self.pulseLine.text()), 
                includeArms = self.armCheck.isChecked())
        
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