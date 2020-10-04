__all__ = [
    "PySprintWarning",
    "DatasetError",
    "InterpolationWarning",
    "FourierWarning",
    "NotCalculatedException",
]


class PySprintWarning(Warning):
    """
    Base pysprint warning class.
    """

    pass


class InterpolationWarning(PySprintWarning):
    """
    This warning is raised when a function applies linear
    interpolation on the data.
    """

    pass


class FourierWarning(PySprintWarning):
    """
    This warning is raised when FFT is called first instead of IFFT.
    Later on it will be improved.
    For more details see help(pysprint.FFTMethod.calculate)
    """

    pass


class DatasetError(Exception):
    """
    This error is raised when invalid type of data encountered
    when initializing a dataset or inherited object.
    """
    pass


class NotCalculatedException(ValueError):
    """
    This error is raised when a function is not available yet
    because something needs to be calculated before.
    """
    pass
