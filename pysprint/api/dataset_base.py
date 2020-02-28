from abc import abstractmethod, ABCMeta

__all__ = ['DatasetBase']

class DatasetBase(metaclass=ABCMeta):
	"""Base metaclass that defines the interface
	for any interferogram."""

	@abstractmethod
	def __init__(self):
		pass

	@property
	@abstractmethod
	def data(self):
		'''The stored dataset.'''
		pass

	@abstractmethod
	def run(self):
		'''Launch the PyQt application with the stored dataset.'''
		pass

	@abstractmethod
	def show(self):
		'''Plot the stored dataset. '''
		pass
