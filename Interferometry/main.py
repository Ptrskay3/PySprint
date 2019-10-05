import sys
from logic import MainProgram
try:
	from PyQt5 import QtWidgets
except ImportError:
	print('PyQt5 is essential for the UI. You can use the API instead.')

def main():
    app = QtWidgets.QApplication(sys.argv)
    my_interface = MainProgram()
    my_interface.showMaximized()
    if my_interface.settings.value('show') == 'True':
    	my_interface.msgbox.exec_()
    	if my_interface.cb.isChecked():
           my_interface.settings.setValue('show', False)
    else:
    	pass
    sys.exit(app.exec_())

if __name__ == "__main__":
  	main()
