from abc import ABC, abstractmethod
import inspect
import warnings

import numpy as np
import matplotlib.pyplot as plt

from pysprint.utils.decorators import _mutually_exclusive_args
from pysprint.utils.decorators import _lazy_property
from pysprint.core._evaluate import gaussian_window
from pysprint.config import _get_config_value
from pysprint.utils import PySprintWarning


class WindowBase(ABC):

    def __init_subclass__(cls, **kwargs):
        cls._get_params()

    @property
    @abstractmethod
    def y(self):
        pass

    @classmethod
    def _get_params(cls):
        init = getattr(cls.__init__, '..', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        POPARGS = ('self', 'center', 'x')

        # introspect the constructor arguments
        init_signature = inspect.signature(init)
        for arg in POPARGS:
            if arg not in init_signature.parameters:
                raise ValueError(f"Window class has no `{arg}` parameter in its __init__ signature,"
                                 f" but it must implement all of the following: {', '.join(POPARGS[1:])}.")

        parameters = [p for p in init_signature.parameters.values()
                      if p.name not in POPARGS 
                      and (p.kind == p.VAR_KEYWORD or p.kind == p.VAR_POSITIONAL)]
        for p in parameters:
            raise RuntimeError("Window classes should always "
                           "specify their parameters in the signature"
                           " of their __init__ (no varargs)."
                           f" {cls} with constructor {init_signature} doesn't "
                           f" follow this convention. Suggestion: don't use *args, **kwargs.")

        # Extract argument names excluding POPARGS
        return sorted([p.name for p in parameters])


    def __init__(self, x, center, **kwargs):
        self.x = x
        self.center = center


    def plot(self, ax=None, scalefactor=1, zorder=90, **kwargs):
        """
        Plot the window.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If not given, plot on the last axis.
        scalefactor : float, optional
            Number describing how much a given window should be scaled up ONLY
            for visibility.
        zorder : float, optional
            The drawing order of artists is determined by their zorder attribute, which is
            a floating point number. Artists with higher zorder are drawn on top. You can
            change the order for individual artists by setting their zorder. The default
            value depends on the type of the Artist.
        """
        if ax is None:
            ax = plt
        ax.plot(self.x, self.y * scalefactor, zorder=zorder, **kwargs)


class GaussianWindow(WindowBase):
    """
    Basic class that implements functionality related to Gaussian
    windows with caching the y values.
    """
    @_mutually_exclusive_args('fwhm', 'std')
    def __init__(self, x, center, fwhm=None, std=None, order=2):
        self.x = x
        self.center = center
        self.fwhm = fwhm
        self.order = order
        self.fwhm = fwhm
        self.std = std
        if self.fwhm is None:
            self.fwhm = self.std * 2 * np.log(2) ** (1 / self.order)

    @_lazy_property
    def y(self):
        """
        The y values of the given window. It's a "lazy_property".
        """
        if not np.min(self.x) <= self.center <= np.max(self.x):
            warnings.warn(
                f"Creating window with center {self.center}, which is outside of the dataset's"
                f" range (from {np.min(self.x):.3f} to {np.max(self.x):.3f}).",
                PySprintWarning
            )
        return gaussian_window(self.x, self.center, self.fwhm, self.order)

    def __str__(self):
        precision = _get_config_value("precision")
        return f"Window(center={self.center:.{precision}f}, fwhm={self.fwhm}, order={self.order})"
