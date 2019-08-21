from ui import Ui_Interferometry
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pandas as pd
from mplwidget import MplWidget
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, 
QDialog, QPushButton, QVBoxLayout, QComboBox, QCheckBox, QLabel,QAction, qApp, QTextEdit, QSpacerItem, QSizePolicy,QHBoxLayout, QGroupBox, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QIcon, QCursor
from datetime import datetime
from evaluate import minMaxMethod, PMCFFMethod, FFT, cutWithGaussian, gaussianWindow  #, IFFT
from help import Help
from smoothing import savgolFilter, findPeaks, convolution, interpolateData, cutData
from loadingData import readData

class mainProgram(QtWidgets.QMainWindow, Ui_Interferometry):
    samX = np.array([])
    samY = np.array([])
    refX = np.array([])
    refY = np.array([])
    a = np.array([])
    b = np.array([])
    temp = np.array([])

    def __init__(self, parent=None):
        super(mainProgram, self).__init__(parent)
        self.setupUi(self)
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
        self.doIFFT.clicked.connect(self.testt)
        self.actionAbout.triggered.connect(self.openHelp)
        self.actionSave_current_data.triggered.connect(self.saveLoadedData)
        self.actionSave_log_file.triggered.connect(self.saveOutput)
        self.actionExit.triggered.connect(self.close)

    def openHelp(self):
        self.window1 = QMainWindow()
        self.ui = Help()
        self.ui.setupUi(self.window1)
        self.window1.show()

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

#dolgozni kell még rajta
    def gaussianCutFunction(self):
        if self.gaussianCut.text() == '':
            self.gaussianCut.setText('100')
        if self.gaussianCut2.text() == '':
            self.gaussianCut2.setText('40')
        if len(self.a)>0 and len(self.b)>0:
            xx = cutWithGaussian(self.b, spike= float(self.gaussianCut.text()), sigma = float(self.gaussianCut2.text()))
            # xx = cutWithGaussian(self.b, spike = 1,sigma = float(self.gaussianCut2.text()))
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.grid()
            self.MplWidget.canvas.axes.plot(self.a, np.abs(xx))
            self.MplWidget.canvas.axes.set_ylabel("Intensity")
            # self.MplWidget.canvas.axes.set_xlabel("Time")
            self.MplWidget.canvas.draw()



    def fftHandler(self):
        if len(self.a)>0 and len(self.b)>0:
            # self.temp = self.a
            self.a, self.b = FFT(self.a, self.b)
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.grid()
            self.MplWidget.canvas.axes.plot(self.a, np.abs(self.b))
            self.MplWidget.canvas.axes.set_ylabel("Intensity")
            self.MplWidget.canvas.axes.set_xlabel("Time")
            self.MplWidget.canvas.draw()
            self.messageOutput('FFT applied to data. Some functions may behave differently. The absolute value is plotted.')
        else:
            self.messageOutput('No data is loaded.')

#átdolgozandó, így nem jó
    # def ifftHandler(self):
    #     if len(self.a)>0 and len(self.b)>0 and len(self.temp)>0:
    #         iffX, iffY = IFFT(self.a, self.b, self.temp)
    #         self.MplWidget.canvas.axes.clear()
    #         self.MplWidget.canvas.axes.grid()
    #         self.MplWidget.canvas.axes.plot(iffX, iffY)
    #         self.MplWidget.canvas.axes.set_ylabel("Intensity")
    #         # self.MplWidget.canvas.axes.set_xlabel("Time")
    #         self.MplWidget.canvas.draw()

    @waitingEffects
    def swapAxes(self):
        self.tableWidget.setRowCount(0)
        if len(self.a)>0:
            self.temp = self.a
            self.a = self.b
            self.b = self.temp
            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.grid()
            self.MplWidget.canvas.axes.plot(self.a, np.abs(self.b))
            self.MplWidget.canvas.axes.set_ylabel("Intensity")
            self.MplWidget.canvas.draw()
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
                    self.MplWidget.canvas.axes.plot(t, np.abs(w))
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
                    self.MplWidget.canvas.axes.plot(t, np.abs(w))
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
                self.MplWidget.canvas.axes.plot(Xdata, np.abs(Ydata))
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
            self.MplWidget.canvas.axes.plot(Xdata, np.abs(Ydata))
            self.MplWidget.canvas.axes.set_ylabel("Intensity")
            # self.MplWidget.canvas.axes.set_xlabel("Angular frequency")
            self.MplWidget.canvas.axes.grid()
            self.MplWidget.canvas.draw()
            
    @pyqtSlot(int)
    def referenceArmClicked(self, refX, refY):
        options = QFileDialog.Options()
        referenceName, _ = QFileDialog.getOpenFileName(None,"Reference arm spectrum", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if referenceName:
                self.refX, self.refY= np.loadtxt(referenceName, usecols=(0,1), unpack = True, delimiter =',')
        except:
            self.messageOutput('Failed')
    
    @pyqtSlot(int)   
    def sampleArmClicked(self, samX, samY):
        options = QFileDialog.Options()       
        sampleName, _ = QFileDialog.getOpenFileName(None,"Sample arm spectrum", "","All Files (*);;Text Files (*.txt)", options=options)
        try:
            if sampleName:
                self.samX, self.samY= np.loadtxt(sampleName, usecols=(0,1), unpack = True, delimiter = ',')
          
        except:
            self.messageOutput('Failed')

    @pyqtSlot(int) 
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
                # self.logOutput.insertPlainText('Calculating..\n')
                cFF = PMCFFMethod(self.a, self.b ,self.refY, self.samY, 
                    p0=[1,1,1, float(self.initGD.text()), float(self.initGDD.text()), float(self.initTOD.text()), float(self.initFOD.text()),
                    float(self.initQOD.text())], showGraph = False)
                labels = ['GD', 'GDD', 'TOD', 'FOD', 'QOD']
                self.logOutput.insertPlainText('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+ ':')
                self.messageOutput('Using Cosine function fit method..')
                try:
                    for item in range(len(cFF)):
                        self.logOutput.insertPlainText(' '+ labels[item] +' =  ' + str(cFF[item]) +'  1/fs^'+str(item+1)+'\n')
                    self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())
                except Exception as e:
                    self.messageOutput('You might need to provide initial guess for parameters.')
            except Exception as e:
                self.messageOutput('You might need to provide initial guess for parameters.')

        if self.methodWidget.currentIndex() == 0 or self.methodWidget.currentIndex() == 3:
            self.logOutput.insertPlainText('\n' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ':')
            self.logOutput.insertPlainText('\n Not implemented yet.\n')
            self.logOutput.verticalScrollBar().setValue(self.logOutput.verticalScrollBar().maximum())
            

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

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = mainProgram()
    gui.show()
    sys.exit(app.exec_())