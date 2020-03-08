from abc import abstractmethod, ABCMeta

__all__ = ['DatasetBase']

C_LIGHT = 299.792458

class DatasetBase(metaclass=ABCMeta):
	"""Base metaclass that defines the interface
	for any interferogram."""

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def GD_lookup(self, *args, **kwargs):
		''' Quick GD lookup'''
		pass

	@property
	@abstractmethod
	def data(self):
		'''The stored dataset.'''
		pass

	@abstractmethod
	def show(self):
		'''Plot the stored dataset. '''
		pass
