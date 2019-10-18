__all__ = ['ImportModel']

import numpy as np
from pysprint.core.loading import read_data

class InvalidCommand(Exception):
	pass


class ImportModel(object):
	def __init__(self, x, y, ref, sam):
		self.d = {}
		self.col1 = x
		self.col2 = y
		self.col3 = ref
		self.col4 = sam
		if len(self.col3) == 0 or len(self.col4) == 0:
			self._ncols = 2
		else:
			self._ncols = 4
		self.create_dict()

	def create_dict(self):
		for i in range(self._ncols):
			self.d['chdomain($'+str(i+1)+')'] = '$' + str(i+1) + ' = np.pi*2*299.793/$' + str(i+1) 
			self.d['$'+str(i+1)] = 'self.col'  + str(i+1)
		self.d['^'] = '**'

	def exec_command(self, com):
		for command, executable in self.d.items():
		    com = com.replace(command, executable)
		try:
			exec(com)
		except SyntaxError:
			raise InvalidCommand

	def unpack(self):
		return self.col1, self.col2, self.col3, self.col4