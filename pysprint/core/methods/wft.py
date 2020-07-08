from functools import lru_cache, wraps

from pysprint.core.methods._fft import FFTMethod
from pysprint.core.evaluate import gaussian_window, cut_gaussian


# https://stackoverflow.com/a/54487188/11751294

def mutually_exclusive(keyword, *keywords):
    keywords = (keyword,) + keywords
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if sum(k in keywords for k in kwargs) != 1:
                raise TypeError('You must specify exactly one of {}'.format(' and '.join(keywords)))
            return func(*args, **kwargs)
        return inner
    return wrapper

def lazy_property(f):
    return property(lru_cache()(f))

class Window:
    """
    Basic class that implements functionality related to Gaussian
    windows.
    """
    def __init__(self, x, center, fwhm, order=2):
        self.x = x
        self.center = center
        self.fwhm = fwhm
        self.order = order

    @lazy_property
    def y(self):
        return gaussian_window(self.x, self.center, self.fwhm, self.order)

    @classmethod
    def from_std(cls, x, center, std, order=2):
        _fwhm = std * 2 * np.log(2) ** (1 / order)
        return cls(x, center, _fwhm, order)

    def plot(self, ax=None, **kwargs):
        """
        Plot the window.

        Parameters
        ----------

        ax : axis
            The axis to plot on. If not given, plot on the last axis.

        **kwargs : 
            keyword arguments to pass to matplotlib.pyplot.plot
        """
        if ax is None:
            ax = plt
        ax.plot(self.x, self.y, **kwargs)



class WFTMethod(FFTMethod):
    """Basic interface for Windowed Fourier Transfrom Method."""

    window_seq = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @mutually_exclusive('std', 'fwhm')
    def add_window(self, center, std=None, fwhm=None, order=2):
        if std:
            window = Window.from_std(self.x, center=center, std=std, order=order)
        else:
            window = Window(self.x, center=center, fwhm=fwhm, order=order)
        self.window_seq[center] = window

    @property
    def windows(self):
        return self.window_seq

    @mutually_exclusive('std', 'fwhm')
    def add_window_sequence(self, start, stop, step, std, fwhm, order):
        """
        Build a window sequence of given parameters to apply on ifg.
        """
        pass
        
    def view_windows(self):
        """
        Gives a rough view of the different windows along with the ifg.
        """
        pass

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    w = WFTMethod(np.arange(100), np.arange(100))
    w.add_window(center=50, order=2)