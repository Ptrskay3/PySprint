from abc import abstractmethod, ABCMeta

__all__ = ['DatasetBase']

class DatasetBase(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self):
		pass

	@property
	@abstractmethod
	def data(self):
		pass

	@abstractmethod
	def run(self):
		pass