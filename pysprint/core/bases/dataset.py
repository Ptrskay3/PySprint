"""
This file implements the basic Dataset class.
"""
import json  # for pretty printing dict
import warnings
from collections.abc import Iterable
from inspect import signature
from contextlib import suppress
from textwrap import dedent
from math import factorial
from copy import copy, deepcopy
import numbers
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysprint.core.bases.dataset_base import DatasetBase, C_LIGHT
from pysprint.mpl_tools.spp_editor import SPPEditor
from pysprint.utils import MetaData, find_nearest
from pysprint.mpl_tools.normalize import DraggableEnvelope

from pysprint.core.preprocess import (
    savgol,
    find_peak,
    convolution,
    cut_data,
    cwt,
)

from pysprint.utils.exceptions import (
    InterpolationWarning,
    DatasetError,
    PySprintWarning,
)

__all__ = ["Dataset"]


class Dataset(metaclass=DatasetBase):
    """Base class for the evaluating methods."""

    meta = MetaData("""Additional info about the dataset""", copy=False)

    def __init__(self, x, y, ref=None, sam=None, meta=None):

        super().__init__()

        self.x = x
        self.y = y
        if ref is None:
            self.ref = []
        else:
            self.ref = ref
        if sam is None:
            self.sam = []
        else:
            self.sam = sam
        self._is_normalized = False
        if not isinstance(self.x, np.ndarray):
            try:
                self.x = np.array(self.x).astype(float)
            except ValueError:
                raise DatasetError("Invalid type of data")
        if not isinstance(self.y, np.ndarray):
            try:
                self.y = np.array(self.y).astype(float)
            except ValueError:
                raise DatasetError("Invalid type of data")
        if not isinstance(self.ref, np.ndarray):
            try:
                self.ref = np.array(self.ref).astype(float)
            except ValueError:
                pass  # just ignore invalid arms
        if not isinstance(self.sam, np.ndarray):
            try:
                self.sam = np.array(self.sam).astype(float)
            except ValueError:
                pass  # just ignore invalid arms

        if len(self.ref) == 0:
            self.y_norm = self.y
            self._is_normalized = self._ensure_norm()

        else:
            self.y_norm = (self.y - self.ref - self.sam) / (
                    2 * np.sqrt(self.sam * self.ref)
            )
            self._is_normalized = True

        self.plotwidget = plt # TODO : Rewrite plotting
        self.xmin = None
        self.xmax = None
        self.probably_wavelength = None
        self._check_domain()

        if meta is not None:
            self.meta = meta

        self._delay = None
        self._positions = None

        self._dispersion_array = None

    def __call__(self, reference_point, *, order=None, show_graph=None):
        """
        Alias for self.calculate.
        """

        if hasattr(self, "calculate"):

            sig = len(signature(self.calculate).parameters)
            if sig == 1:
                if order or show_graph:
                    warnings.warn("order and show_graph has no effect here.")
                self.calculate(reference_point)
            elif sig == 3:

                # set up defaults here
                if order is None:
                    order = 3
                if show_graph is None:
                    show_graph = False

                self.calculate(
                    reference_point=reference_point, order=order, show_graph=show_graph
                )

            else:
                raise ValueError("Unknown function signature.")
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def chrange(self):
        pass

    #  TODO : The plot must be formatted.
    def phase_plot(self, exclude_GD=False):
        """
        Plot the phase if the dispersion is already calculated.

        Parameters
        ----------

        exclude_GD : bool
            Whether to exclude the GD part of the polynomial.
            Default is `False`.
        """
        if not np.all(self._dispersion_array):
            raise ValueError("Dispersion must be calculated before plotting the phase.")

        coefs = np.array(
            [
                self._dispersion_array[i] / factorial(i + 1)
                for i in range(len(self._dispersion_array))
            ]
        )

        if exclude_GD:
            coefs[0] = 0

        phase_poly = np.poly1d(coefs[::-1], r=False)

        self.plotwidget.plot(self.x, phase_poly(self.x))
        self.plotwidget.grid()
        self.plotwidget.ylabel(r"$\Phi\, [rad]$")
        self.plotwidget.xlabel(r"$\omega \,[PHz]$")
        self.plotwidget.show()

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        if isinstance(value, numbers.Number):
            if value < np.min(self.x) or value > np.max(self.x):
                raise ValueError(
                    f"Cannot set SPP position to {value} since it's not in the dataset's range."
                )
        # TODO: maybe we don't need to distinguish between np.ndarray and Iterable
        elif isinstance(value, np.ndarray) or isinstance(value, Iterable):
            for val in value:
                if not isinstance(val, numbers.Number):
                    raise ValueError(
                        f"Expected numeric values, got {type(val)} instead."
                    )
                if val < np.min(self.x) or val > np.max(self.x):
                    raise ValueError(
                        f"Cannot set SPP position to {val} since it's not in the dataset's range."
                    )
        self._positions = value

    def _ensure_norm(self):
        """
        Ensure the interferogram is normalized and only a little part
        which is outlying from the [-1, 1] interval (because of noise).
        """
        idx = np.where((self.y_norm > 2))
        val = len(idx[0]) / len(self.y_norm)
        if val > 0.015:  # this is a custom threshold, which often works..
            return False
        return True

    def scale_up(self):
        """
        If the interferogram is normalized to [0, 1] interval, scale
        up to [-1, 1] with easy algebra.. Just in case you need comparison,
        or any other purpose.
        """
        self.y_norm = (self.y_norm - 0.5) * 2
        self.y = (self.y - 0.5) * 2

    def GD_lookup(self, reference_point=2.355, engine="cwt", silent=False, **kwargs):
        """
        Quick GD lookup: it finds extremal points near the
        `reference_point` and returns an average value of 2*np.pi
        divided by distances between consecutive minimal or maximal values.
        Since it's relying on peak detection, the results may be irrelevant
        in some cases. If the parent class is `~pysprint.CosFitMethod`, then
        it will set the predicted value as initial parameter for fitting.

        Parameters
        ----------

        reference_point : float
            The reference point for the algorithm.

        engine : str
            The backend to use. Must be "cwt", "normal" or "fft".
            "cwt" will use `scipy.signal.find_peaks_cwt` function to
            detect peaks, "normal" will use `scipy.signal.find_peaks`
            to detect peaks. The "fft" engine uses Fourier-transform and
            looks for the outer peak to guess delay value. It's not
            reliable when working with low delay values.

        silent : bool
            Whether to print the results immediately. Default in `False`.

        **kwargs
            Additional keyword arguments to pass for peak detection
            algorithms. These are:
                pmin, pmax, threshold, width, floor_thres, etc..
            Most of them are described in the `find_peaks` and
            `find_peaks_cwt` docs.
        """

        # TODO: implement FFT-based engine

        if engine == "fft":
            warnings.warn(
                "FFT based engine is not implemented yet, falling back to 'cwt'."
            )
            engine = "cwt"

        if engine not in ("cwt", "normal"):
            raise ValueError("Engine must be `cwt` or `normal`.")

        if engine == "cwt":
            width = kwargs.pop("width", 35)
            floor_thres = kwargs.pop("floor_thres", 0.05)
            x_min, _, x_max, _ = self.detect_peak_cwt(
                width=width, floor_thres=floor_thres
            )

            # just validation
            _ = kwargs.pop("pmin", 0.1)
            _ = kwargs.pop("pmax", 0.1)
            _ = kwargs.pop("threshold", 0.35)

        else:
            pmin = kwargs.pop("pmin", 0.1)
            pmax = kwargs.pop("pmax", 0.1)
            threshold = kwargs.pop("threshold", 0.35)
            x_min, _, x_max, _ = self.detect_peak(
                pmin=pmin, pmax=pmax, threshold=threshold
            )

            # just validation
            _ = kwargs.pop("width", 10)
            _ = kwargs.pop("floor_thres", 0.05)

        if kwargs:
            raise TypeError(f"Invalid argument:{kwargs}")

        try:
            closest_val, idx1 = find_nearest(x_min, reference_point)
            m_closest_val, m_idx1 = find_nearest(x_max, reference_point)
        except ValueError:
            if not silent:
                print("Prediction failed.\nSkipping.. ")
            return
        try:
            truncated = np.delete(x_min, idx1)
            second_closest_val, _ = find_nearest(truncated, reference_point)
        except IndexError:
            if not silent:
                print("Prediction failed.\nSkipping.. ")
            return
        try:
            m_truncated = np.delete(x_max, m_idx1)
            m_second_closest_val, _ = find_nearest(m_truncated, reference_point)
        except IndexError:
            if not silent:
                print("Prediction failed.\nSkipping.. ")
            return

        lowguess = 2 * np.pi / np.abs(closest_val - second_closest_val)
        highguess = 2 * np.pi / np.abs(m_closest_val - m_second_closest_val)

        #  estimate the GD with that
        if hasattr(self, "params"):
            self.params[3] = (lowguess + highguess) / 2

        if not silent:
            print(
                f"The predicted GD is ± {((lowguess + highguess) / 2):.5f} fs"
                f" based on reference point of {reference_point}."
            )

    def _safe_cast(self):
        """
        Return a copy of key attributes in order to prevent
        inplace modification.
        """
        x, y, ref, sam = (
            np.copy(self.x),
            np.copy(self.y),
            np.copy(self.ref),
            np.copy(self.sam),
        )
        return x, y, ref, sam

    @staticmethod
    def wave2freq(value):
        """Switches values between wavelength and angular frequency."""
        return (2 * np.pi * C_LIGHT) / value

    _dispatch = wave2freq.__func__

    @staticmethod
    def freq2wave(value):
        """Switches values between angular frequency and wavelength."""
        return Dataset._dispatch(value)

    def _check_domain(self):
        """
        Checks the domain of data just by looking at x axis' minimal value.
        Units are obviously not added yet, we work in nm and PHz...
        """
        if min(self.x) > 50:
            self.probably_wavelength = True
        else:
            self.probably_wavelength = False

    # TODO : Make a better metadata parser.

    @classmethod
    def parse_raw(
            cls, basefile, ref=None, sam=None, skiprows=8, decimal=",", sep=";", meta_len=5,
    ):
        """
        Dataset object alternative constructor.
        Helps to load in data just by giving the filenames in
        the target directory.

        Parameters:
        ----------
        basefile: `str`
            base interferogram
            file generated by the spectrometer

        ref: `str`, optional
            reference arm's spectra
            file generated by the spectrometer

        sam: `str`, optional
            sample arm's spectra
            file generated by the spectrometer

        skiprows: `int`, optional
            Skip rows at the top of the file. Default is `8`.

        sep: `str`, optional
            The delimiter in the original interferogram file.
            Default is `;`.

        decimal: `str`, optional
            Character recognized as decimal separator in the original dataset.
            Often `,` for European data.
            Default is `,`.

        meta_len: `int`, optional
            The first `n` lines in the original file containing the meta
            information about the dataset. It is parsed to be dict-like.
            If the parsing fails, a new entry will be created in the
            dictionary with key `unparsed`.
            Default is `5`.
        """
        if skiprows < meta_len:
            warnings.warn(
                f"Skiprows is currently {skiprows}, but"
                " meta information is set to {meta_len}"
                " lines. This implies that either one is probably wrong.",
                PySprintWarning,
            )
            meta_len = skiprows

        with open(basefile) as file:
            comm = next(file).strip("\n").split("-")[-1].lstrip(" ")
            additional = (
                next(file).strip("\n").strip("\x00").split(":")
                for _ in range(1, meta_len)
            )
            if meta_len != 0:
                cls.meta = {"comment": comm}
            try:
                for info in additional:
                    cls.meta[info[0]] = info[1].strip()
            except IndexError:
                cls.meta["unparsed"] = str(list(additional))
        df = pd.read_csv(
            basefile,
            skiprows=skiprows,
            sep=sep,
            decimal=decimal,
            usecols=[0, 1],
            names=["x", "y"],
        )
        if ref is not None and sam is not None:
            r = pd.read_csv(
                ref,
                skiprows=skiprows,
                sep=sep,
                decimal=decimal,
                usecols=[0, 1],
                names=["x", "y"],
            )
            s = pd.read_csv(
                sam,
                skiprows=skiprows,
                sep=sep,
                decimal=decimal,
                usecols=[0, 1],
                names=["x", "y"],
            )
            return cls(df["x"].values, df["y"].values, r["y"].values, s["y"].values)
        return cls(df["x"].values, df["y"].values)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if isinstance(self._delay, np.ndarray):
            pprint_delay = self._delay.flat[0]
        elif isinstance(self._delay, Iterable):
            pprint_delay = next((_ for _ in self._delay), None)
        elif isinstance(self._delay, numbers.Number):
            pprint_delay = self._delay
        else:
            pprint_delay = "-"
        string = dedent(
            f"""
        {type(self).__name__}
        ----------
        Parameters
        ----------
        Datapoints: {len(self.x)}
        Predicted domain: {'wavelength' if self.probably_wavelength else 'frequency'}
        Range: from {np.min(self.x):.2f} to {np.max(self.x):.2f} {'nm' if self.probably_wavelength else 'PHz'}
        Normalized: {self._is_normalized}
        Delay value: {(str(pprint_delay) + ' fs') if np.all(self._delay) else 'Not given'}
        SPP position(s): {str(self._positions) + ' PHz' if np.all(self._positions) else 'Not given'}
        ----------------------------
        Metadata extracted from file
        ----------------------------
        {json.dumps(self.meta, indent=4)}"""
        )
        return re.sub('^\s+', '', string, flags=re.MULTILINE)

    @property
    def data(self):
        """
        Returns the *current* dataset as `pandas.DataFrame`.
        """
        if self._is_normalized:
            try:
                self._data = pd.DataFrame(
                    {
                        "x": self.x,
                        "y": self.y,
                        "sample": self.sam,
                        "reference": self.ref,
                        "y_normalized": self.y_norm,
                    }
                )
            except ValueError:
                self._data = pd.DataFrame({"x": self.x, "y": self.y})
        else:
            self._data = pd.DataFrame({"x": self.x, "y": self.y})
        return self._data

    #  https://stackoverflow.com/a/15774013/11751294
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def is_normalized(self):
        """Retuns whether the dataset is normalized."""
        return self._is_normalized

    def chdomain(self, inplace=True):
        """
        Changes from wavelength [nm] to ang. freq. [PHz]
        domain and vica versa.

        Parameters
        ----------

        inplace : bool
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.
        """
        if inplace:
            self.x = (2 * np.pi * C_LIGHT) / self.x
            self._check_domain()
            if hasattr(self, "original_x"):
                self.original_x = self.x
        else:
            obj = copy(self)
            obj.chdomain(inplace=True)
            return obj

    def detect_peak_cwt(self, width, floor_thres=0.05):
        x, y, ref, sam = self._safe_cast()
        xmax, ymax, xmin, ymin = cwt(
            x, y, ref, sam, width=width, floor_thres=floor_thres
        )
        return xmax, ymax, xmin, ymin

    def savgol_fil(self, window=5, order=3):
        """
        Applies Savitzky-Golay filter on the dataset.

        Parameters:
        ----------
        window: int
            Length of the convolutional window for the filter.
            Default is `10`.

        order: int
            Degree of polynomial to fit after the convolution.
            If not odd, it's incremented by 1. Must be lower than window.
            Usually it's a good idea to stay with a low degree, e.g 3 or 5.
            Default is 3.

        Notes:
        ------
        If arms were given, it will merge them into the `self.y` and
        `self.y_norm` variables. Also applies a linear interpolation o
        n dataset (and raises warning).
        """
        self.x, self.y_norm = savgol(
            self.x, self.y, self.ref, self.sam, window=window, order=order
        )
        self.y = self.y_norm
        self.ref = []
        self.sam = []
        warnings.warn(
            "Linear interpolation have been applied to data.", InterpolationWarning,
        )

    def slice(self, start=None, stop=None, inplace=True):
        """
        Cuts the dataset on x axis.

        Parameters:
        ----------
        start : float
            start value of cutting interval
            Not giving a value will keep the dataset's original minimum value.
            Note that giving `None` will leave original minimum untouched too.
            Default is `None`.

        stop : float
            stop value of cutting interval
            Not giving a value will keep the dataset's original maximum value.
            Note that giving `None` will leave original maximum untouched too.
            Default is `None`.

        inplace : bool
            Whether to apply the operation on the dataset in an "inplace" manner.
            This means if inplace is True it will apply the changes directly on
            the current dataset and returns None. If inplace is False, it will
            leave the current object untouched, but returns a copy of it, and
            the operation will be performed on the copy. It's useful when
            chaining operations on a dataset.

        Notes:
        ------

        If arms were given, it will merge them into the `self.y` and
        `self.y_norm` variables. After this operation, the arms' spectra
        cannot be retrieved.
        """
        if inplace:
            self.x, self.y_norm = cut_data(
                self.x, self.y, self.ref, self.sam, start=start, stop=stop
            )
            self.ref = []
            self.sam = []
            self.y = self.y_norm
            # Just to make sure it's correctly shaped. Later on we might
            # delete this.
            if hasattr(self, "original_x"):
                self.original_x = self.x
            self._is_normalized = self._ensure_norm()
        else:
            obj = copy(self)
            obj.slice(start=start, stop=stop, inplace=True)
            return obj

    def convolution(self, window_length, std=20):
        """
        Applies a convolution with a gaussian on the dataset.

        Parameters:
        ----------
        window_length: int
            Length of the gaussian window.

        std: float
            Standard deviation of the gaussian window.
            Default is `20`.


        Returns
        -------

        None

        Notes:
        ------
        If arms were given, it will merge them into the `self.y` and
        `self.y_norm` variables.
        Also applies a linear interpolation on dataset.
        """
        self.x, self.y_norm = convolution(
            self.x, self.y, self.ref, self.sam, window_length, standev=std
        )
        self.ref = []
        self.sam = []
        self.y = self.y_norm
        warnings.warn(
            "Linear interpolation have been applied to data.", InterpolationWarning,
        )

    def detect_peak(self, pmax=0.1, pmin=0.1, threshold=0.1, except_around=None):
        """
        Basic algorithm to find extremal points in data
        using ``scipy.signal.find_peaks``.

        Parameters:
        ----------

        pmax : float
            Prominence of maximum points.
            The lower it is, the more peaks will be found.
            Default is `0.1`.

        pmin : float
            Prominence of minimum points.
            The lower it is, the more peaks will be found.
            Default is `0.1`.

        threshold : float
            Sets the minimum distance (measured on y axis) required for a
            point to be accepted as extremal.
            Default is 0.

        except_around : interval (array or tuple),
            Overwrites the threshold to be 0 at the given interval.
            format is `(lower, higher)` or `[lower, higher]`.
            Default is None.

        Returns:
        -------
        xmax: `array-like`
            x coordinates of the maximums

        ymax: `array-like`
            y coordinates of the maximums

        xmin: `array-like`
            x coordinates of the minimums

        ymin: `array-like`
            y coordinates of the minimums
        """
        x, y, ref, sam = self._safe_cast()
        xmax, ymax, xmin, ymin = find_peak(
            x,
            y,
            ref,
            sam,
            pro_max=pmax,
            pro_min=pmin,
            threshold=threshold,
            except_around=except_around,
        )
        return xmax, ymax, xmin, ymin

    def _plot_SPP_if_valid(self, **kwargs):
        """
        Mark SPPs on the plot if they are valid.
        """
        if isinstance(self.positions, numbers.Number):
            x_closest, idx = find_nearest(self.x, self.positions)
            try:
                self.plotwidget.plot(x_closest, self.y_norm[idx], **kwargs)
            except Exception:  # TODO: handle that exception precisely, maybe ValueError?
                self.plotwidget.plot(x_closest, self.y[idx], **kwargs)

        if isinstance(self.positions, np.ndarray) or isinstance(
                self.positions, Iterable
        ):
            for i, val in enumerate(self.positions):
                x_closest, idx = find_nearest(self.x, self.positions[i])
                try:
                    self.plotwidget.plot(x_closest, self.y_norm[idx], **kwargs)
                except Exception:  # TODO: handle that exception precisely, maybe ValueError?
                    self.plotwidget.plot(x_closest, self.y[idx], **kwargs)

    # TODO: Add **kwargs
    def show(self):
        """
        Draws a graph of the current dataset using matplotlib.
        """
        if np.iscomplexobj(self.y):
            self.plotwidget.plot(self.x, np.abs(self.y))
        else:
            try:
                self.plotwidget.plot(self.x, self.y_norm, "r")
            except Exception:  # TODO: handle that exception precisely, maybe ValueError?
                self.plotwidget.plot(self.x, self.y, "r")
        self._plot_SPP_if_valid(color="black", marker="o", markersize=10)
        self.plotwidget.grid()
        self.plotwidget.show(block=True)

    def normalize(self, filename=None, smoothing_level=0):
        """
        Normalize the interferogram by finding upper and lower envelope
        on an interactive matplotlib editor.

        Parameters
        ----------

        filename : `str`
            Save the normalized interferogram named by filename in the
            working directory. If not given it will not be saved.
            Default None.

        smoothing_level : int
            The smoothing level used on the dataset before finding the
            envelopes. It applies Savitzky-Golay filter under the hood.

        Returns
        -------
        None
        """
        x, y, _, _ = self._safe_cast()
        if smoothing_level != 0:
            x, y = savgol(x, y, [], [], window=smoothing_level)
        _l_env = DraggableEnvelope(x, y, "l")
        y_transform = _l_env.get_data()
        _u_env = DraggableEnvelope(x, y_transform, "u")
        y_final = _u_env.get_data()
        self.y = y_final
        self.y_norm = y_final
        self._is_normalized = True
        self.plotwidget.title("Final")
        self.show()
        if filename:
            if not filename.endswith(".txt"):
                filename += ".txt"
            np.savetxt(filename, np.transpose([self.x, self.y]), delimiter=",")
            print(f"Successfully saved as {filename}")

    def open_SPP_panel(self):
        """
        Opens the interactive matplotlib editor for SPP data.
        Use `i` button to add a new point, use `d` key to delete one.
        """
        _spp = SPPEditor(self.x, self.y_norm)
        self.delay, self.positions = _spp.get_data()

    def emit(self):
        """
        Emit the current SPP data.

        Returns
        -------

        delay : np.ndarray
        The delay value for the current dataset, shaped exactly like
        positions.

        positions : np.ndarray
        The given SPP positions.
        """
        if self.positions is None:
            raise ValueError("SPP positions are missing.")
        if self.delay is None:
            raise ValueError("Delay value is missing.")
        # validate if it's typed by hand..
        if not isinstance(self._positions, np.ndarray):
            self._positions = np.asarray(self.positions)
        if not isinstance(self.delay, np.ndarray):
            self.delay = np.ones_like(self.positions) * self.delay
        return self.delay, self.positions

    def set_SPP_data(self, delay, positions, force=False):
        """
        Set the SPP data (delay and SPP positions) for the dataset

        Parameters
        ----------

        delay : float
            The delay value that belongs to the current interferogram.
            Must be given in `fs` units.

        positions : float or iterable
            The SPP positions that belong to the current interferogram.
            Must be float or sequence of floats (tuple, list, np.ndarray, etc.)

        force : bool, optional
            Can be used to set specific SPP positions which are outside of
            the dataset's range. Note that in most cases you should avoid using
            this option. Default is `False`.

        Returns
        -------
        None

        Notes
        -----
        Every position given must be in the current dataset's range, otherwise
        `ValueError` is raised. Be careful to change domain to frequency before
        feeding values into this function.
        """
        if not isinstance(delay, float):
            delay = float(delay)
        delay = np.array(np.ones_like(positions) * delay)
        self.delay = delay
        if force:
            with suppress(ValueError):
                self._positions = positions
        else:
            self.positions = positions
