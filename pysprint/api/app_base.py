import sys
from pysprint.api.dataset_base import DatasetBase

C_LIGHT = 299.792458

class BaseApp(DatasetBase):
	def __init__(self):
		super().__init__()

	def run(self):
		"""
		Opens up the GUI with the loaded data.
		"""
		from pysprint.logic import MainProgram
		try:
			from PyQt5 import QtWidgets
		except ImportError:
			print('PyQt5 is essential for the UI. Use the API instead.')
		print('Building up UI..')
		app = QtWidgets.QApplication(sys.argv)
		main_app = MainProgram()
		main_app.showMaximized()
		main_app.a = self.x
		main_app.b = self.y
		main_app.samY = self.sam
		main_app.refY = self.ref
		if main_app.settings.value('show') == 'True':
			main_app.msgbox.exec_()
			if main_app.cb.isChecked():
				main_app.settings.setValue('show', False)
		main_app.redraw_graph()
		main_app.fill_table()
		main_app.track_stats()
		sys.exit(app.exec_())
		
	@property
	def data(self):
		raise NotImplementedError

	def show(self):
		raise NotImplementedError
