import os
import contextlib
import warnings

import numpy as np
from scipy.fftpack import fftshift

from pysprint.core.bases.dataset import Dataset
from pysprint.core.ffts_non_uniform import nuifft
from pysprint.core.evaluate import (
    fft_method,
    cut_gaussian,
    ifft_method,
    args_comp,
    gaussian_window,
)
from pysprint.utils.exceptions import FourierWarning
from pysprint.core.ffts_auto import _run
from pysprint.core.phase import Phase

__all__ = ["FFTMethod"]


class FFTMethod(Dataset):
    """
    Basic interface for the Fourier transform method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #  making sure it's not normalized
        if self._is_normalized:
            self.y_norm = self.y
            self._is_normalized = False
        self.original_x = self.x
        self.at = None
        self.std = None
        self.window_order = None
        self._ifft_called_first = False
        self.phase = None

    def shift(self, axis="x"):
        """
        Equivalent to scipy.fftpack.fftshift, but it's easier to
        use this function instead, because we don't need to explicitly
        call the class' x and y attribute.

        Parameter(s):
        ------------
        axis: str, default is 'x'
            either 'x', 'y' or 'both'
        """
        if axis == "x":
            self.x = fftshift(self.x)
        elif axis == "y":
            self.y = fftshift(self.y)
        elif axis == "both" or axis == 'xy' or axis == 'yx':
            self.y = fftshift(self.y)
            self.x = fftshift(self.x)
        else:
            raise ValueError("axis should be either `x`, `y` or `both`.")

    def ifft(
        self, interpolate=True, usenifft=False, eps=1e-12, exponent="positive"
    ):
        """
        Applies ifft to the dataset.

        Parameter(s):
        ------------

        interpolate: bool, default is True
            Whether to apply linear interpolation on the dataset
            before transforming.

        usenifft: bool, default is False
            Whether to use non unifrom fft

        """
        self._ifft_called_first = True
        if usenifft:
            x_spaced = np.linspace(self.x[0], self.x[-1], len(self.x))
            timestep = np.diff(x_spaced)[0]
            x_axis = np.fft.fftfreq(len(self.x), d=timestep / (2 * np.pi))
            y_transform = nuifft(
                self.x,
                self.y,
                gl=len(self.x),
                df=(x_axis[1] - x_axis[0]),
                epsilon=eps,
                exponent=exponent,
            )
            self.x, self.y = x_axis, np.fft.fftshift(y_transform)

        else:
            self.x, self.y = ifft_method(
                self.x, self.y, interpolate=interpolate
            )

    def fft(self):
        """
        Applies fft to the dataset.
        If ifft was not called first, inaccurate results might happen.
        It will be fixed later on.
        Check calculate function's docstring for more detail.
        """
        if not self._ifft_called_first:
            warnings.warn(
                "This module is designed to call ifft before fft",
                FourierWarning
            )
        self.x, self.y = fft_method(self.original_x, self.y)

    def window(self, at, fwhm, window_order=6, plot=True):
        """
        Draws a gaussian window on the plot with the desired parameters.
        The maximum value is adjusted for the dataset mostly for
        visibility reasons. You should explicitly call self.show()
        after this function is set.

        Parameters:
        ----------

        at: float
            maximum of the gaussian curve

        fwhm: float
            Full width at half maximum of the gaussian

        window_order: int, default is 6
            Order of the gaussian curve.
            If not even, it's incremented by 1.
        """
        self.at = at
        self.fwhm = fwhm
        self.window_order = window_order
        gaussian = gaussian_window(
            self.x, self.at, self.fwhm, self.window_order
        )
        self.plotwidget.plot(self.x, gaussian * max(abs(self.y)), "r--")
        if plot:
            self.show()

    def apply_window(self):
        """
        If window function is correctly set, applies changes to the dataset.
        """
        self.plotwidget.clf()
        self.plotwidget.cla()
        self.plotwidget.close()
        self.y = cut_gaussian(
            self.x,
            self.y,
            spike=self.at,
            fwhm=self.fwhm,
            win_order=self.window_order,
        )

    def calculate(self, reference_point, order, show_graph=False):
        """
        FFTMethod's calculate function.

        Parameters:
        ----------

        reference_point: float
            reference point on x axis

        fit_order: int
            Polynomial (and maximum dispersion) order to fit. Must be in [1,5].

        show_graph: bool, optional
            shows a the final graph of the spectral phase and fitted curve.

        Returns:
        -------

        dispersion: array-like
            [GD, GDD, TOD, FOD, QOD]

        dispersion_std: array-like
            standard deviations due to uncertanity of the fit
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

        fit_report: lmfit report
            if lmfit is available, the fit report

        Notes:
        ------

        Decorated with print_disp, so the results are immediately
        printed without explicitly saying so.

        Currently the x-axis transformation is sloppy, because we cache the
        original x axis and not transforming it	backwards.
        In addition we need to keep track of interpolation and
        zero-padding too. Currently the transforms are correct only if
        first ifft was used. For now it's doing okay: giving good results.
        For consistency we should still implement that a better way later.
        """
        dispersion, dispersion_std, fit_report = args_comp(
            self.x,
            self.y,
            ref_point=reference_point,
            fit_order=order,
            show_graph=show_graph,
        )
        self._dispersion_array = dispersion
        return dispersion, dispersion_std, fit_report

    def autorun(
        self,
        reference_point=None,
        order=None,
        *,
        enable_printing=True,
        skip_domain_check=False,
        only_phase=False,
        show_graph=True,
        usenifft=False,
    ):

        if not reference_point or not order:
            only_phase = True

        if not enable_printing:

            with open(os.devnull, "w") as g, contextlib.redirect_stdout(g):
                _run(
                    self,
                    skip_domain_check=skip_domain_check,
                    show_graph=show_graph,
                    usenifft=usenifft,
                )
            if only_phase:
                y = np.unwrap(np.angle(self.y), axis=0)
                self.phase = Phase(self.x, y)
                self.phase.plot()
                return self.x, y
            self.calculate(
                reference_point=reference_point, order=order, show_graph=True
            )
        else:
            _run(
                self,
                skip_domain_check=skip_domain_check,
                show_graph=show_graph,
                usenifft=usenifft,
            )
            if only_phase:
                y = np.unwrap(np.angle(self.y), axis=0)
                self.phase = Phase(self.x, y)
                self.phase.plot()
                return self.x, y
            self.calculate(
                reference_point=reference_point, order=order, show_graph=True
            )
