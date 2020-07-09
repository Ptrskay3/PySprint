from functools import lru_cache, wraps

import numpy as np

from pysprint.core.methods._fft import FFTMethod
from pysprint.core.evaluate import gaussian_window, cut_gaussian


# https://stackoverflow.com/a/54487188/11751294

def mutually_exclusive_args(keyword, *keywords):
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
    def __init__(self, x, center, fwhm, order=2, **kwargs):
        self.x = x
        self.center = center
        self.fwhm = fwhm
        self.order = order
        self.mpl_style = kwargs

    @lazy_property
    def y(self):
        return gaussian_window(self.x, self.center, self.fwhm, self.order)

    @classmethod
    def from_std(cls, x, center, std, order=2):
        _fwhm = std * 2 * np.log(2) ** (1 / order)
        return cls(x, center, _fwhm, order)

    def plot(self, ax=None):
        """
        Plot the window.

        Parameters
        ----------

        ax : axis
            The axis to plot on. If not given, plot on the last axis.
        """
        if ax is None:
            ax = plt
        ax.plot(self.x, self.y, **self.mpl_style)


class WFTMethod(FFTMethod):
    """Basic interface for Windowed Fourier Transfrom Method."""

    window_seq = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @mutually_exclusive_args('std', 'fwhm')
    def add_window(self, center, std=None, fwhm=None, order=2, **kwargs):
        if std:
            window = Window.from_std(self.x, center=center, std=std, order=order, **kwargs)
        else:
            window = Window(self.x, center=center, fwhm=fwhm, order=order, **kwargs)
        self.window_seq[center] = window

    @property
    def windows(self):
        return self.window_seq

    def add_window_arange(self, start, stop, step, std=None, fwhm=None, order=2, **kwargs):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.arange.
        """
        arr = np.arange(start, stop, step)
        if std:
            for cent in arr:
                self.add_window(cent, std=std, order=order, **kwargs)
        else:
            for cent in arr:
                self.add_window(cent, fwhm=fwhm, order=order, **kwargs)
        

    @mutually_exclusive_args("std", "fwhm")
    def add_window_linspace(self, start, stop, num, std=None, fwhm=None, order=2, **kwargs):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.linspace.
        """
        arr = np.linspace(start, stop, num)
        if std:
            for cent in arr:
                self.add_window(center=cent, std=std, order=order, **kwargs)
        else:
            for cent in arr:
                self.add_window(center=cent, fwhm=fwhm, order=order, **kwargs)

    @mutually_exclusive_args("std", "fwhm")
    def add_window_geomspace(self, start, stop, num, std=None, fwhm=None, order=2, **kwargs):
        """
        Build a window sequence of given parameters to apply on ifg.
        Works similar to numpy.geomspace.
        """
        arr = np.geomspace(start, stop, num)
        if std:
            for cent in arr:
                self.add_window(center=cent, std=std, order=order, **kwargs)
        else:
            for cent in arr:
                self.add_window(center=cent, fwhm=fwhm, order=order, **kwargs)

    def view_windows(self, ax=None):
        """
        Gives a rough view of the different windows along with the ifg.
        """
        # TODO: check for too crowded image later and warn user.
        # Maybe just display a sample that case?
        for _, val in self.window_seq.items():
            val.plot(ax=ax)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w = WFTMethod(np.arange(100), np.arange(100))
    w.add_window_linspace(20, 70, 100, fwhm=50)
    w.view_windows()
    plt.legend()
    plt.show(block=True)
    print(w)