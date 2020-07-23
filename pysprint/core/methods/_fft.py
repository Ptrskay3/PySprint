import os
import contextlib
import warnings
from copy import copy

import numpy as np
from scipy.fftpack import fftshift

from pysprint.core.bases.dataset import Dataset
from pysprint.core.nufft import nuifft
from pysprint.utils.exceptions import FourierWarning
from pysprint.core.fft_tools import _run
from pysprint.core.phase import Phase
from pysprint.core.evaluate import (
    fft_method,
    cut_gaussian,
    ifft_method,
    args_comp,
    gaussian_window,
)

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
        self.fwhm = None
        self.window_order = None
        self.phase = None
        self._ifft_called_first = False
        self.nufft_used = False

    def shift(self, axis, inplace=True):
        """
        Equivalent to `scipy.fftpack.fftshift`, but it's easier to
        use this function instead, because we don't need to explicitly
        call the class' x and y attribute.

        Parameters
        ----------
        axis : str
            either 'x', 'y', 'both', 'xy' or 'yx'.

        inplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.
        """
        if inplace:
            if axis == "x":
                self.x = fftshift(self.x)
            elif axis == "y":
                self.y = fftshift(self.y)
            elif axis == "both" or axis == "xy" or axis == "yx":
                self.y = fftshift(self.y)
                self.x = fftshift(self.x)
            else:
                raise ValueError("axis should be either `x`, `y` or `both`.")
        else:
            obj = copy(self)
            obj.shift(axis=axis, inplace=True)
            return obj

    def ifft(
        self,
        interpolate=True,
        usenifft=False,
        eps=1e-12,
        exponent="positive",
        inplace=True,
    ):
        """
        Applies inverse Fast Fourier Transfrom to the dataset.

        Parameters
        ----------

        interpolate : bool, default is True -- WILL BE REMOVED
            Whether to apply linear interpolation on the dataset
            before transforming.

        usenifft : bool, optional
            Whether to use non uniform fft. It uses the algorithm
            described in the references. This means the interferogram
            will *not* be linearly interpolated. Default is False.

        eps : float, optional
            The desired approximate error for the non uniform FFT result. Must be
            in range 1E-33 < eps < 1E-1, though be aware that the errors are
            only well calibrated near the range 1E-12 ~ 1E-6. Default is 1E-12.

        exponent : str, optional
            if 'negative', compute the transform with a negative exponent.
            if 'positive', compute the transform with a positive exponent.
            Default is `positive`.

        inplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.

        Notes
        -----

        The basic scheme is ifft -> windowing -> fft, so you should call
        these functions in this order. Otherwise the transforms may be
        inconsistent.

        If numba is not installed the non uniform FTT is approximately
        5x times slower, but still remains comparable to `np.fft.ifft`.

        References
        ----------

        [1] Dutt A., Rokhlin V. : Fast Fourier Transforms for Nonequispaced Data II,
            Applied and Computational Harmonic Analysis
            Volume 2, Issue 1, January 1995, Pages 85-100
            (1995)

        [2] Greengard, Leslie & Lee, June-Yub.: Accelerating the
            Nonuniform Fast Fourier Transform,
            Society for Industrial and Applied Mathematics.
            46. 443-454. 10.1137/S003614450343200X.
            (2004)
        """
        self.nufft_used = usenifft

        if inplace:
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
                self.x, self.y = ifft_method(self.x, self.y, interpolate=interpolate)
        else:
            obj = copy(self)
            obj.ifft(
                interpolate=interpolate,
                usenifft=usenifft,
                eps=eps,
                exponent=exponent,
                inplace=True,
            )
            return obj

    def fft(self, inplace=True):
        """
        Applies fft to the dataset.
        If ifft was not called first, inaccurate results might happen.

        Parameters
        ----------

        inplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.
        """
        if inplace:
            if not self._ifft_called_first:
                warnings.warn(
                    "This module is designed to call ifft before fft", FourierWarning
                )
            self.x, self.y = fft_method(self.original_x, self.y)
        else:
            obj = copy(self)
            obj.fft(inplace=True)
            return obj

    def window(self, at, fwhm, window_order=6, plot=True, inplace=True):
        """
        Draws a gaussian window on the plot with the desired parameters.
        The maximum value is adjusted for the dataset's maximum value,
        mostly for visibility.

        Parameters:
        ----------

        at : float
            The maximum of the gaussian curve.

        fwhm : float
            Full width at half maximum of the gaussian

        window_order : int, optional
            Order of the gaussian curve.
            If not even, it's incremented by 1.
            Default is 6.

        plot : bool, optional
            Whether to immediately show the window with the data.
            Default is `True`.

        inplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.
        """
        if inplace:
            self.at = at
            self.fwhm = fwhm
            self.window_order = window_order
            gaussian = gaussian_window(self.x, self.at, self.fwhm, self.window_order)
            self.plt.plot(self.x, gaussian * max(abs(self.y)), "r--")
            if plot:
                self.show()
        else:
            obj = copy(self)
            obj.window(
                at=at, fwhm=fwhm, window_order=window_order, plot=plot, inplace=True
            )
            return obj

    def apply_window(self, inplace=True):
        """
        If window function is set, applies window on the dataset.

        inplace : bool, optional
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.
        """
        if inplace:
            self.plt.clf()
            self.plt.cla()
            self.plt.close()
            self.y = cut_gaussian(
                self.x,
                self.y,
                spike=self.at,
                fwhm=self.fwhm,
                win_order=self.window_order,
            )
        else:
            obj = copy(self)
            obj.apply_window(inplace=True)
            return obj

    def retrieve_phase(self):
        """
        Retrieve *only the phase* after the transforms. This will
        unwrap the angles and constructs a pysprint.core.phase.Phase object.

        Returns
        -------
        phase : pysprint.core.phase.Phase
            The phase object. See its docstring for more info.
        """
        if self.nufft_used:
            self.shift("y")
        y = np.unwrap(np.angle(self.y), axis=0)
        self.phase = Phase(self.x, y)
        return self.phase  # because of inplace ops. we need to return the phase

    def calculate(self, reference_point, order, show_graph=False):
        """
        FFTMethod's calculate function. It will unwrap the phase by changing
        deltas between values to 2*pi complement. After that, fit a curve to
        determine dispersion coefficients.

        Parameters:
        ----------

        reference_point : float
            The reference point on the x axis.

        order : int
            Polynomial (and maximum dispersion) order to fit. Must be in [1, 5].

        show_graph : bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.

        Returns:
        -------

        dispersion : array-like
            The dispersion coefficients in the form of:
            [GD, GDD, TOD, FOD, QOD]

        dispersion_std : array-like
            Standard deviations due to uncertainty of the fit.
            It is only calculated if lmfit is installed. The form is:
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]

        fit_report : str
            If lmfit is available returns the fit report, else returns an
            empty string.

        Notes:
        ------

        Decorated with print_disp, so the results are immediately
        printed without explicitly saying so.

        Developer commentary:
        Currently the x-axis transformation is sloppy, because we cache the
        original x axis and not transforming it	backwards.
        In addition we need to keep track of interpolation and
        zero-padding too. Currently the transforms are correct only if
        first ifft was used. For now it's doing okay: giving good results.
        For consistency we should still implement that a better way later.
        """
        self.retrieve_phase()
        dispersion, dispersion_std, fit_report = self.phase._fit(
            reference_point=reference_point, order=order
        )
        if show_graph:
            self.phase.plot()

        self._dispersion_array = dispersion
        return -dispersion, dispersion_std, fit_report

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
        """
        Automatically run the Fourier Transfrom based evaluation on the dataset.
        It's not as reliable as I want it to be, so use it carefully. I'm working
        on making it as competent and useful as possible.

        Parameters
        ----------

        reference_point : float, optional
            The reference point on the x axis. If not given, only_phase mode
            will be activated. Default is None.

        order : int, optional
            Polynomial (and maximum dispersion) order to fit. Must be in [1, 5].
            If not given, only_phase mode will be activated. Default is None.

        only_phase : bool, optional
            If True, activate the only_phase mode, which will retrieve the phase
            without fitting a curve, and return a `pysprint.core.Phase.phase` object.
            Default is False (also not giving enough information for curve fitting
            will automatically activate it).

        enable_printing : bool, optional
            If True enable printing the detailed results. Default is True.

        skip_domain_check : bool, optional
            If True skip the interferogram domain check and force the algorithm
            to perform actions without changing domain. If False, check for potential
            wrong domains and change for an appropriate one. Default is False.

        show_graph : bool, optional
            If True show the graph with the phase and the fitted curve, if there is any.
            Default is True.

        usenifft : bool, optional
            If True use the Non Uniform Fast Fourier Transform algorithm. For more details
            see `help(pysprint.FFTMethod.ifft)`. Default is False.

        References
        ----------

        [1] Dutt A., Rokhlin V. : Fast Fourier Transforms for Nonequispaced Data II,
            Applied and Computational Harmonic Analysis
            Volume 2, Issue 1, January 1995, Pages 85-100
            (1995)

        [2] Greengard, Leslie & Lee, June-Yub.: Accelerating the
            Nonuniform Fast Fourier Transform,
            Society for Industrial and Applied Mathematics.
            46. 443-454. 10.1137/S003614450343200X.
            (2004)
        """
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
                return self.phase
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
                return self.phase
            self.calculate(
                reference_point=reference_point, order=order, show_graph=True
            )

    def errorplot(self, *args, **kwargs):
        try:
            getattr(self.phase, "errorplot", None)(*args, **kwargs)
        except TypeError:
            raise ValueError("Must calculate before plotting errors.")

    @property
    def get_phase(self):
        if self.phase is not None:
            return self.phase
        raise ValueError("Must retrieve the phase first.")

    @property
    def errors(self):
        return getattr(self.phase, "errors", None)
