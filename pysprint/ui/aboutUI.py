# -*- coding: utf-8 -*-
"""
Help window
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QMessageBox, QPushButton


class Help(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Help")
        MainWindow.setFixedSize(801, 530)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(310, 460, 160, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.textArea = QtWidgets.QTextBrowser(self.centralwidget)
        self.textArea.setGeometry(QtCore.QRect(40, 20, 721, 431))
        self.textArea.setReadOnly(True)
        self.textArea.setObjectName("textArea")
        self.textArea.insertPlainText('Interferogram Evaluator ')
        self.textArea.insertPlainText('created by Péter Leéh ')
        self.textArea.insertPlainText('(2019)\n\n')
        self.textArea.insertPlainText('For more detalis visit: ')
        self.textArea.insertHtml("<a href='https://github.com/Ptrskay3/PySprint'>GitHub</a>\n  ")
        self.textArea.textCursor().insertHtml('\n\n or send an email to <b>leeh123peter@gmail.com</b>')
        self.textArea.setOpenExternalLinks(True)
        self.exbtn = QPushButton('Close', self.centralwidget)
        self.exbtn.move(350,460)
        self.exbtn.clicked.connect(self.close)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 801, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Help"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Help()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

