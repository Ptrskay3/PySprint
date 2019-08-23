import sys
from logic import mainProgram
from PyQt5 import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    my_interface = mainProgram()
    my_interface.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
  	main()
