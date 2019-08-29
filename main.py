import sys
from logic import MainProgram

try:
	from PyQt5 import QtWidgets
except ImportError:
	print('PyQt5 is essential to run this program.')


def main():
    app = QtWidgets.QApplication(sys.argv)
    my_interface = MainProgram()
    my_interface.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
  	main()
