"""
This file implements the Dataset class with all the functionality
that an interferogram should have in general.
"""
import base64
from collections.abc import Iterable
from contextlib import suppress, contextmanager
from io import BytesIO
import json
import logging
from math import factorial
import numbers
import re
from textwrap import dedent
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from jinja2 import Template

from pysprint.config import _get_config_value
from pysprint.core.bases._dataset_base import _DatasetBase
from pysprint.core.bases._dataset_base import C_LIGHT
from pysprint.core.bases._apply import _DatasetApply
from pysprint.core._evaluate import is_inside
from pysprint.core._evaluate import ifft_method
from pysprint.core._fft_tools import find_center
from pysprint.core.io._parser import _parse_raw
from pysprint.mpl_tools.spp_editor import SPPEditor
from pysprint.mpl_tools.normalize import DraggableEnvelope
from pysprint.utils import MetaData, find_nearest
from pysprint.utils.decorators import inplacify
from pysprint.core._preprocess import (
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

logger = logging.getLogger(__name__)
FORMAT = "[ %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

__all__ = ["Dataset"]


class Dataset(metaclass=_DatasetBase):
    """
    This class implements all the functionality a dataset
    should have in general.
    """

    meta = MetaData("""Additional info about the dataset""", copy=False)

    def __init__(
            self,
            x,
            y,
            ref=None,
            sam=None,
            meta=None,
            errors="raise",
            callback=None,
            parent=None,
            **kwargs
    ):
        """
        Base constructor for Dataset.

        Parameters
        ----------
        x : np.ndarray
            The x values.
        y : np.ndarray
            The y values.
        ref : np.ndarray, optional
            The reference arm's spectra.
        sam : np.ndarray, optional
            The sample arm's spectra.
        meta : dict-like
            The dictionary containing further information about the dataset.
            Can be extended, or set to be any valid ~collections.abc.Mapping.
        errors: str, optional
            Whether to raise on missmatching sized data. Must be "raise" or
            "force". If "force" then truncate to the shortest size. Default is
            "raise".
        callback : callable, optional
            The function that notifies parent objects about SPP related
            changes. In most cases the user should leave this empty. The
            default callback is only initialized if this object is constructed
            by the `pysprint.SPPMethod` object.
        parent : any class, optional
            The object which handles the callback function. In most cases
            the user should leave this empty.
        kwargs : dict, optional
            The window class to use in WFTMethod. Has no effect while using other
            methods. Must be a subclass of pysprint.core.windows.WindowBase.

        Note
        ----
        To load in data by files, see the other constructor `parse_raw`.
        """
        super().__init__()

        if errors not in ("raise", "force"):
            raise ValueError("errors must be `raise` or `force`.")

        self.callback = callback or (lambda *args: args)
        self.parent = parent

        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
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

        if not len(self.x) == len(self.y):
            if errors == 'raise':
                raise ValueError(
                    f"Mismatching data shapes with {self.x.shape} and {self.y.shape}."
                )
            else:
                truncated_shape = min(len(self.x), len(self.y))
                # probably we should cut down the first half
                self.x, self.y = self.x[-truncated_shape:], self.y[-truncated_shape:]

        if len([x for x in (self.ref, self.sam) if len(x) != 0]) == 1:
            warnings.warn(
                "Reference and sample arm should be passed together or neither one.",
                PySprintWarning
            )

        if len(self.ref) == 0 or len(self.sam) == 0:
            self.y_norm = self.y
            self._is_normalized = self._ensure_norm()

        else:
            if not np.all([len(self.sam) == len(self.x), len(self.ref) == len(self.x)]):
                if errors == 'raise':
                    raise ValueError(
                        f"Mismatching data shapes with {self.x.shape}, "
                        f"{self.ref.shape} and {self.sam.shape}."
                    )
                else:
                    truncated_shape = min(len(self.x), len(self.ref), len(self.sam), len(self.y))
                    # same as above..
                    self.ref, self.sam = self.ref[-truncated_shape:], self.sam[-truncated_shape:]

            self.y_norm = (self.y - self.ref - self.sam) / (
                    2 * np.sqrt(self.sam * self.ref)
            )
            self._is_normalized = True

        self.plt = plt
        self.xmin = None
        self.xmax = None
        self.probably_wavelength = None
        self.unit = None
        self._check_domain()

        if meta is not None:
            self.meta = meta

        self._delay = None
        self._positions = None

        nanwarning = np.isnan(self.y_norm).sum()
        infwarning = np.isinf(self.y_norm).sum()
        if nanwarning > 0 or infwarning > 0:
            warnings.warn(
                ("Extreme values encountered during normalization.\n"
                f"Nan values: {nanwarning}\nInf values: {infwarning}"),
                PySprintWarning
            )

        self._dispersion_array = None

    @inplacify
    def chrange(self, current_unit, target_unit="phz"):
        """
        Change the domain range of the dataset.

        Supported units for frequency:
            * PHz
            * THz
            * GHz
        Supported units for wavelength:
            * um
            * nm
            * pm
            * fm

        Parameters
        ----------
        current_unit : str
            The current unit of the domain. Case insensitive.
        target_unit : str, optional
            The target unit. Must be compatible with the currect unit.
            Case insensitive. Default is `phz`.
        """
        current_unit, target_unit = current_unit.lower(), target_unit.lower()
        conversions = {
            "um": {"um": 1, "nm": 1000, "pm": 1E6, "fm": 1E9},
            "nm": {"um": 1 / 1000, "nm": 1, "pm": 1000, "fm": 1E6},
            "pm": {"um": 1 / 1E6, "nm": 1 / 1000, "pm": 1, "fm": 1000},
            "fm": {"um": 1 / 1E9, "nm": 1 / 1E6, "pm": 1 / 1000, "fm": 1},
            "phz": {"phz": 1, "thz": 1000, "ghz": 1E6},
            "thz": {"phz": 1 / 1000, "thz": 1, "ghz": 1000},
            "ghz": {"phz": 1 / 1E6, "thz": 1 / 1000, "ghz": 1}
        }
        try:
            ratio = float(conversions[current_unit][target_unit])
        except KeyError as error:
            raise ValueError("Units are not compatible") from error
        self.x = (self.x * ratio)
        self.unit = self._render_unit(target_unit)
        return self

    def __len__(self):
        return len(self.x)

    @staticmethod
    def _render_unit(unit, mpl=False):
        unit = unit.lower()
        charmap = {
            "um": (r"\mu m", "um"),
            "nm": ("nm", "nm"),
            "pm": ("pm", "pm"),
            "fm": ("fm", "fm"),
            "phz": ("PHz", "PHz"),
            "thz": ("THz", "THz"),
            "ghz": ("GHz", "GHz")
        }
        if mpl:
            return charmap[unit][0]
        return charmap[unit][1]

    @inplacify
    def transform(self, func, axis=None, args=None, kwargs=None):
        """
        Function which enables to apply arbitrary function to the
        dataset.

        Parameters
        ----------
        func : callable
            The function to apply on the dataset.
        axis : int or str, optional
            The axis which is the operation is performed on.
            Must be 'x', 'y', '0' or '1'.
        args : tuple, optional
            Additional arguments to pass to func.
        kwargs : dict, optional
            Additional keyword arguments to pass to func.
        """
        operation = _DatasetApply(
            obj=self, func=func, axis=axis, args=args, kwargs=kwargs
        )
        operation.perform()
        return self

    #  TODO : Rewrite this
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

        self.plt.plot(self.x, phase_poly(self.x))
        self.plt.grid()
        self.plt.ylabel(r"$\Phi\, [rad]$")
        self.plt.xlabel(r"$\omega \,[PHz]$")
        self.plt.show()

    @property
    def delay(self):
        """
        Return the delay value if set.
        """
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value
        try:
            self.callback(self, self.parent)
        except ValueError:
            pass  # delay or position is missing

    @property
    def positions(self):
        """
        Return the SPP position(s) if set.
        """
        return self._positions

    @positions.setter
    def positions(self, value):
        if isinstance(value, numbers.Number):
            if value < np.min(self.x) or value > np.max(self.x):
                raise ValueError(
                    f"Cannot set SPP position to {value} since it's not in the dataset's range."
                )
        # FIXME: maybe we don't need to distinguish between np.ndarray and Iterable
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
        try:
            self.callback(self, self.parent)
        except ValueError:
            pass  # delay or position is missing

    def _ensure_norm(self):
        """
        Ensure the interferogram is normalized and only a little part
        which is outlying from the [-1, 1] interval (because of noise).
        """
        try:
            idx = np.where((self.y_norm > 2))
            val = len(idx[0]) / len(self.y_norm)
        except TypeError as e:
            raise DatasetError("Non-numeric values found while reading dataset.") from e
        if val > 0.015:  # this is a custom threshold, which often works..
            return False
        return True

    def scale_up(self):
        """
        If the interferogram is normalized to [0, 1] interval, scale
        up to [-1, 1] with easy algebra.
        """
        self.y_norm = (self.y_norm - 0.5) * 2
        self.y = (self.y - 0.5) * 2

    def GD_lookup(self, reference_point=None, engine="cwt", silent=False, **kwargs):
        """
        Quick GD lookup: it finds extremal points near the
        `reference_point` and returns an average value of 2*pi
        divided by distances between consecutive minimal or maximal values.
        Since it's relying on peak detection, the results may be irrelevant
        in some cases. If the parent class is `~pysprint.CosFitMethod`, then
        it will set the predicted value as initial parameter for fitting.

        Parameters
        ----------
        reference_point : float
            The reference point for the algorithm.
        engine : str, optional
            The backend to use. Must be "cwt", "normal" or "fft".
            "cwt" will use `scipy.signal.find_peaks_cwt` function to
            detect peaks, "normal" will use `scipy.signal.find_peaks`
            to detect peaks. The "fft" engine uses Fourier-transform and
            looks for the outer peak to guess delay value. It's not
            reliable when working with low delay values.
        silent : bool, optional
            Whether to print the results immediately. Default in `False`.
        kwargs : dict, optional
            Additional keyword arguments to pass for peak detection
            algorithms. These are:
            pmin, pmax, threshold, width, floor_thres, etc..
            Most of them are described in the `find_peaks` and
            `find_peaks_cwt` docs.
        """
        precision = _get_config_value("precision")

        if engine not in ("cwt", "normal", "fft"):
            raise ValueError("Engine must be `cwt`, `fft` or `normal`.")

        if reference_point is None and engine != "fft":
            warnings.warn(
                f"Engine `{engine}` isn't available without reference point, falling back to FFT based prediction.",
                PySprintWarning
            )
            engine = "fft"

        if engine == "fft":
            pred, _ = find_center(*ifft_method(self.x, self.y))
            if pred is None:
                if not silent:
                    print("Prediction failed, skipping.")
                return
            print(f"The predicted GD is ± {pred:.{precision}f} fs.")

            if hasattr(self, "params"):
                self.params[3] = pred
            return

        if engine == "cwt":
            widths = kwargs.pop("widths", np.arange(1, 20))
            floor_thres = kwargs.pop("floor_thres", 0.05)
            x_min, _, x_max, _ = self.detect_peak_cwt(
                widths=widths, floor_thres=floor_thres
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
            _ = kwargs.pop("widths", np.arange(1, 10))
            _ = kwargs.pop("floor_thres", 0.05)

        if kwargs:
            raise TypeError(f"Invalid argument:{kwargs}")

        try:
            closest_val, idx1 = find_nearest(x_min, reference_point)
            m_closest_val, m_idx1 = find_nearest(x_max, reference_point)
        except (ValueError, IndexError):
            if not silent:
                print("Prediction failed, skipping.. ")
            return
        try:
            truncated = np.delete(x_min, idx1)
            second_closest_val, _ = find_nearest(truncated, reference_point)
        except (IndexError, ValueError):
            if not silent:
                print("Prediction failed, skipping.. ")
            return
        try:
            m_truncated = np.delete(x_max, m_idx1)
            m_second_closest_val, _ = find_nearest(m_truncated, reference_point)
        except (IndexError, ValueError):
            if not silent:
                print("Prediction failed, skipping.. ")
            return

        lowguess = 2 * np.pi / np.abs(closest_val - second_closest_val)
        highguess = 2 * np.pi / np.abs(m_closest_val - m_second_closest_val)

        #  estimate the GD with that
        if hasattr(self, "params"):
            self.params[3] = (lowguess + highguess) / 2

        if not silent:
            print(
                f"The predicted GD is ± {((lowguess + highguess) / 2):.{precision}f} fs"
                f" based on reference point of {reference_point:.{precision}f}."
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
        """Switches a single value between wavelength and angular frequency."""
        return (2 * np.pi * C_LIGHT) / value

    _dispatch = wave2freq.__func__

    @staticmethod
    def freq2wave(value):
        """Switches a single value between angular frequency and wavelength."""
        return Dataset._dispatch(value)

    def _check_domain(self):
        """
        Checks the domain of data just by looking at x axis' minimal value.
        Units are obviously not added yet, we work in nm and PHz...
        """
        try:
            if min(self.x) > 50:
                self.probably_wavelength = True
                self.unit = "nm"
            else:
                self.probably_wavelength = False
                self.unit = "PHz"

        # This is the first function to fail if the user sets up
        # wrong values. Usually..
        except TypeError as error:
            msg = ValueError(
                "The file could not be parsed properly."
            )
            raise msg from error

    @classmethod
    def parse_raw(
        cls,
        filename,
        ref=None,
        sam=None,
        skiprows=0,
        decimal=".",
        sep=None,
        delimiter=None,
        comment=None,
        usecols=None,
        names=None,
        swapaxes=False,
        na_values=None,
        skip_blank_lines=True,
        keep_default_na=False,
        meta_len=1,
        errors="raise",
        callback=None,
        parent=None,
        **kwargs
    ):
        """
        Dataset object alternative constructor.
        Helps to load in data just by giving the filenames in
        the target directory.

        Parameters
        ----------
        filename: `str`
            base interferogram
            file generated by the spectrometer
        ref: `str`, optional
            reference arm's spectra
            file generated by the spectrometer
        sam: `str`, optional
            sample arm's spectra
            file generated by the spectrometer
        skiprows: `int`, optional
            Skip rows at the top of the file. Default is `0`.
        decimal: `str`, optional
            Character recognized as decimal separator in the original dataset.
            Often `,` for European data.
            Default is `.`.
        sep: `str`, optional
            The delimiter in the original interferogram file.
            Default is `,`.
        delimiter: `str`, optional
            The delimiter in the original interferogram file.
            This is preferred over the `sep` argument if both given.
            Default is `,`.
        comment: `str`, optional
            Indicates remainder of line should not be parsed. If found at the beginning
            of a line, the line will be ignored altogether. This parameter must be a
            single character. Default is `'#'`.
        usecols: list-like or callable, optional
            If there a multiple columns in the file, use only a subset of columns.
            Default is [0, 1], which will use the first two columns.
        names: array-like, optional
            List of column names to use. Default is ['x', 'y']. Column marked
            with `x` (`y`) will be treated as the x (y) axis. Combined with the
            usecols argument it's possible to select data from a large number of
            columns.
        swapaxes: bool, optional
            Whether to swap x and y values in every parsed file. Default is False.
        na_values: scalar, str, list-like, or dict, optional
            Additional strings to recognize as NA/NaN. If dict passed, specific
            per-column NA values. By default the following values are interpreted as
            NaN: ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’,
            ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’,
            ‘<NA>’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’.
        skip_blank_lines: bool
            If True, skip over blank lines rather than interpreting as NaN values.
            Default is True.
        keep_default_na: bool
            Whether or not to include the default NaN values when parsing the data.
            Depending on whether na_values is passed in, the behavior changes. Default
            is False. More information available at:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        meta_len: `int`, optional
            The first `n` lines in the original file containing the meta
            information about the dataset. It is parsed to be dict-like.
            If the parsing fails, a new entry will be created in the
            dictionary with key `unparsed`.
            Default is `1`.
        errors: string, optional
            Determines the way how mismatching sized datacolumns behave.
            The default is `raise`, and it will raise on any error.
            If set to `force`, it will truncate every array to have the
            same shape as the shortest column. It truncates from
            the top of the file.
        callback : callable, optional
            The function that notifies parent objects about SPP related
            changes. In most cases the user should leave this empty. The
            default callback is only initialized if this object is constructed
            by the `pysprint.SPPMethod` object.
        parent : any class, optional
            The object which handles the callback function. In most cases
            the user should leave this empty.
        kwargs : dict, optional
            The window class to use in WFTMethod. Has no effect while using other
            methods. Must be a subclass of pysprint.core.windows.WindowBase.
        """

        parsed = _parse_raw(
            filename=filename,
            ref=ref,
            sam=sam,
            skiprows=skiprows,
            decimal=decimal,
            sep=sep,
            delimiter=delimiter,
            comment=comment,
            usecols=usecols,
            names=names,
            swapaxes=swapaxes,
            na_values=na_values,
            skip_blank_lines=skip_blank_lines,
            keep_default_na=keep_default_na,
            meta_len=meta_len
        )

        return cls(**parsed, errors=errors, callback=callback, parent=parent, **kwargs)

    def __str__(self):
        _unit = self._render_unit(self.unit)
        precision = _get_config_value("precision")
        string = dedent(
            f"""
        {type(self).__name__}
        ----------
        Parameters
        ----------
        Datapoints: {len(self.x)}
        Predicted domain: {'wavelength' if self.probably_wavelength else 'frequency'}
        Range: from {np.min(self.x):.{precision}f} to {np.max(self.x):.{precision}f} {_unit}
        Normalized: {self._is_normalized}
        Delay value: {str(self._format_delay()) + ' fs' if self._delay is not None else 'Not given'}
        SPP position(s): {str(self._format_positions()) + ' PHz' if np.all(self._positions) else 'Not given'}
        ----------------------------
        Metadata extracted from file
        ----------------------------
        """
        )
        string = re.sub('^\s+', '', string, flags=re.MULTILINE)
        string += json.dumps(self.meta, indent=4, sort_keys=True)
        return string

    def _repr_html_(self):  # TODO: move this to a separate template file
        _unit = self._render_unit(self.unit)
        precision = _get_config_value("precision")
        t = f"""
        <div id="header" class="row" style="height:10%;width:100%;">
        <div style='float:left' class="column">
        <table style="border:1px solid black;float:top;">
        <tbody>
        <tr>
        <td colspan=2 style="text-align:center">
        <font size="5">{type(self).__name__}</font>
        </td>
        </tr>
        <tr>
        <td colspan=2 style="text-align:center">
        <font size="3.5">Parameters</font>
        </td>
        </tr>
        <tr>
        <td style="text-align:center"><b>Datapoints<b></td>
            <td style="text-align:center"> {len(self.x)}</td>
        </tr>
        <tr>
            <td style="text-align:center"><b>Predicted domain<b> </td>
            <td style="text-align:center"> {'wavelength' if self.probably_wavelength else 'frequency'} </td>
        </tr>
        <tr>
        <td style="text-align:center"> <b>Range min</b> </td>
        <td style="text-align:center">{np.min(self.x):.{precision}f} {_unit}</td>
        </tr>
        <tr>
        <td style="text-align:center"> <b>Range max</b> </td>
        <td style="text-align:center">{np.max(self.x):.{precision}f} {_unit}</td>
        </tr>
        <tr>
        <td style="text-align:center"> <b>Normalized</b></td>
        <td style="text-align:center"> {self._is_normalized} </td>
        </tr>
        <tr>
        <td style="text-align:center"><b>Delay value</b></td>
        <td style="text-align:center">{str(self._format_delay()) + ' fs' if self._delay is not None else 'Not given'}</td>
        </tr>
        <tr>
        <td style="text-align:center"><b>SPP position(s)</b></td>
        <td style="text-align:center">{str(self._format_positions()) + ' PHz' if np.all(self._positions) else 'Not given'}</td>
        </tr>
        <tr>
        <td colspan=2 style="text-align:center">
        <font size="3.5">Metadata</font>
        </td>
        </tr>
        """
        jjstring = Template("""
        {% for key, value in meta.items() %}
           <tr>
        <th style="text-align:center"> <b>{{ key }} </b></th>
        <td style="text-align:center"> {{ value }} </td>
           </tr>
        {% endfor %}
            </tbody>
        </table>
        </div>
        <div style='float:leftt' class="column">""")
        rendered_fig = self._render_html_plot()
        return t + jjstring.render(meta=self.meta) + rendered_fig

    def _render_html_plot(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        self.plot(ax=ax)
        plt.close()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html_fig = "<img src=\'data:image/png;base64,{}\'>".format(encoded)
        return html_fig

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

    # from : https://stackoverflow.com/a/15774013/11751294
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    @property
    def is_normalized(self):
        """Retuns whether the dataset is normalized."""
        return self._is_normalized

    @inplacify
    def chdomain(self):
        """
        Changes from wavelength [nm] to ang. freq. [PHz]
        domain and vica versa.
        """
        self.x = (2 * np.pi * C_LIGHT) / self.x
        self._check_domain()
        if hasattr(self, "original_x"):
            self.original_x = self.x
        return self

    def detect_peak_cwt(self, widths, floor_thres=0.05, side="both"):
        """
        Basic algorithm to find extremal points in data
        using ``scipy.signal.find_peaks_cwt``.

        Parameters
        ----------
        widths : np.ndarray
            The widths passed to `find_peaks_cwt`.
        floor_thres : float
            Will be removed.
        side : str
            The side to use. Must be "both", "max" or "min".
            Default is "both".

        Returns
        -------
        xmax : `array-like`
            x coordinates of the maximums
        ymax : `array-like`
            y coordinates of the maximums
        xmin : `array-like`
            x coordinates of the minimums
        ymin : `array-like`
            y coordinates of the minimums

        Note
        ----
        When using "min" or "max" as side, all the detected minimal and
        maximal values will be returned, but only the given side will be
        recorded for further calculation.
        """
        if side not in ("both", "min", "max"):
            raise ValueError("Side must be 'both', 'min' or 'max'.")

        if hasattr(self, "_is_onesided"):
            self._is_onesided = side != "both"

        x, y, ref, sam = self._safe_cast()
        xmax, ymax, xmin, ymin = cwt(
            x, y, ref, sam, widths=widths, floor_thres=floor_thres
        )
        self.xmax = xmax
        self.xmin = xmin

        if side == "both":
            self.xmax = xmax
            self.xmin = xmin

        elif side == "min":
            self.xmin = xmin

        elif side == "max":
            self.xmax = xmax

        logger.info(f"{len(xmax)} max values and {len(xmin)} min values were found.")
        return xmax, ymax, xmin, ymin

    def savgol_fil(self, window=5, order=3):
        """
        Applies Savitzky-Golay filter on the dataset.

        Parameters
        ----------
        window : int
            Length of the convolutional window for the filter.
            Default is `10`.
        order : int
            Degree of polynomial to fit after the convolution.
            If not odd, it's incremented by 1. Must be lower than window.
            Usually it's a good idea to stay with a low degree, e.g 3 or 5.
            Default is 3.

        Note
        ----
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

    @inplacify
    def slice(self, start=None, stop=None):
        """
        Cuts the dataset on x axis.

        Parameters
        ----------
        start : float
            Start value of cutting interval.
            Not giving a value will keep the dataset's original minimum value.
            Note that giving `None` will leave original minimum untouched too.
            Default is `None`.
        stop : float
            Stop value of cutting interval.
            Not giving a value will keep the dataset's original maximum value.
            Note that giving `None` will leave original maximum untouched too.
            Default is `None`.

        Note
        ----
        If arms were given, it will merge them into the `self.y` and
        `self.y_norm` variables. After this operation, the arms' spectra
        cannot be retrieved.
        """
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
        return self

    def convolution(self, window_length, std=20):
        """
        Convolve the dataset with a specified Gaussian window.

        Parameters
        ----------
        window_length : int
            Length of the gaussian window.
        std : float
            Standard deviation of the gaussian window.
            Default is `20`.

        Note
        ----
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

    @inplacify
    def resample(self, N, kind="linear", **kwds):
        """
        Resample the interferogram to have `N` datapoints.

        Parameters
        ----------
        N : int
            The number of datapoints required.
        kind : str, optional
            The type of interpolation to use. Default is `linear`.
        kwds : optional
            Additional keyword argument to pass to `scipy.interpolate.interp1d`.

        Raises
        ------
        PySprintWarning, if trying to subsample to lower `N` datapoints than original.
        """
        f = interp1d(self.x, self.y_norm, kind, **kwds)
        if N < len(self.x):
            N = len(self.x)
            warnings.warn(
                "Trying to resample to lower resolution, keeping shape..", PySprintWarning
            )
        xnew = np.linspace(np.min(self.x), np.max(self.x), N)
        ynew = f(xnew)
        setattr(self, "x", xnew)
        setattr(self, "y_norm", ynew)
        return self

    def detect_peak(
        self, pmax=0.1, pmin=0.1, threshold=0.1, except_around=None, side="both"
    ):
        """
        Basic algorithm to find extremal points in data
        using ``scipy.signal.find_peaks``.

        Parameters
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
        side : str
            The side to use. Must be "both", "max" or "min".
            Default is "both".

        Returns
        -------
        xmax : `array-like`
            x coordinates of the maximums
        ymax : `array-like`
            y coordinates of the maximums
        xmin : `array-like`
            x coordinates of the minimums
        ymin : `array-like`
            y coordinates of the minimums

        Note
        ----
        When using "min" or "max" as side, all the detected minimal and
        maximal values will be returned, but only the given side will be
        recorded for further calculation.
        """
        if side not in ("both", "min", "max"):
            raise ValueError("Side must be 'both', 'min' or 'max'.")

        if hasattr(self, "_is_onesided"):
            self._is_onesided = side != "both"

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
        if side == "both":
            self.xmax = xmax
            self.xmin = xmin

        elif side == "min":
            self.xmin = xmin

        elif side == "max":
            self.xmax = xmax

        logger.info(f"{len(xmax)} max values and {len(xmin)} min values were found.")
        return xmax, ymax, xmin, ymin

    def _plot_SPP_if_valid(self, ax=None, **kwargs):
        """
        Mark SPPs on the plot if they are valid.
        """
        if ax is None:
            ax = self.plt
        if isinstance(self.positions, numbers.Number):
            if is_inside(self.positions, self.x):
                x_closest, idx = find_nearest(self.x, self.positions)
                try:
                    ax.plot(x_closest, self.y_norm[idx], **kwargs)
                except (ValueError, TypeError):
                    ax.plot(x_closest, self.y[idx], **kwargs)

        if isinstance(self.positions, np.ndarray) or isinstance(
                self.positions, Iterable
        ):
            if np.array(self.positions).ndim == 0:
                self.positions = np.atleast_1d(self.positions)
            # iterate over 0-d array: need to cast np.atleast_1d
            for i, val in enumerate(self.positions):
                if is_inside(self.positions[i], self.x):
                    x_closest, idx = find_nearest(self.x, self.positions[i])
                    try:
                        ax.plot(x_closest, self.y_norm[idx], **kwargs)
                    except (ValueError, TypeError):
                        ax.plot(x_closest, self.y[idx], **kwargs)

    def _format_delay(self):
        if self.delay is None:
            return ""
        if isinstance(self.delay, np.ndarray):
            if self.delay.size == 0:
                return 0
            delay = np.atleast_1d(self.delay).flatten()
            return delay[0]
        elif isinstance(self.delay, (list, tuple)):
            return self.delay[0]
        elif isinstance(self.delay, numbers.Number):
            return self.delay
        elif isinstance(self.delay, str):
            try:
                delay = float(self.delay)
            except ValueError as e:
                raise TypeError("Delay value not understood.") from e
            return delay
        else:
            raise TypeError("Delay value not understood.")

    def _format_positions(self):
        if self.positions is None:
            return "Not given"
        if isinstance(self.positions, np.ndarray):
            positions = np.atleast_1d(self.positions).flatten()
            return ", ".join(map(str, positions))
        elif isinstance(self.positions, (list, tuple)):
            return ", ".join(map(str, self.positions))
        elif isinstance(self.positions, numbers.Number):
            return self.positions
        elif isinstance(self.positions, str):
            split = self.positions.split(",")
            try:
                positions = [float(p) for p in split]
            except ValueError as e:
                raise TypeError("Delay value not understood.") from e
            return ", ".join(map(str, positions))
        else:
            raise TypeError("Delay value not understood.")

    def _prepare_SPP_data(self):
        pos_x, pos_y = [], []
        if self.positions is not None:
            position = np.array(self.positions, dtype=np.float64).flatten()
            for i, val in enumerate(position):
                if is_inside(position[i], self.x):
                    x_closest, idx = find_nearest(self.x, position[i])
                    try:
                        pos_x.append(x_closest)
                        pos_y.append(self.y_norm[idx])
                    except (ValueError, TypeError):
                        pos_x.append(x_closest)
                        pos_y.append(self.y[idx])
            pos_x = np.array(pos_x)
            pos_y = np.array(pos_y)
        return pos_x, pos_y

    # TODO: Remove the duplicated logic. This function is in pysprint's init.py
    # and we can't circular import it. It should be moved to a separate file.
    def plot_outside(self, *args, **kwargs):
        """
        Plot the current dataset out of the notebook. For detailed
        parameters see `Dataset.plot` function.
        """
        backend = kwargs.pop("backend", "Qt5Agg")
        original_backend = plt.get_backend()
        try:
            plt.switch_backend(backend)
            self.plot(*args, **kwargs)
            plt.show(block=True)
        except (AttributeError, ImportError, ModuleNotFoundError) as err:
            raise ValueError(
                f"Couldn't set backend {backend}, you should manually "
                "change to an appropriate GUI backend. (Matplotlib 3.3.1 "
                "is broken. In that case use backend='TkAgg')."
            ) from err
        finally:
            plt.switch_backend(original_backend)

    def plot(self, ax=None, title=None, xlim=None, ylim=None, **kwargs):
        """
        Plot the dataset.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An axis to draw the plot on. If not given, it will plot
            on the last used axis.
        title : str, optional
            The title of the plot.
        xlim : tuple, optional
            The limits of x axis.
        ylim : tuple, optional
            The limits of y axis.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot function.

        Note
        ----
        If SPP positions are correctly set, it will mark them on plot.
        """
        datacolor = kwargs.pop("color", "red")
        nospp = kwargs.pop("nospp", False)
        _unit = self._render_unit(self.unit, mpl=True)
        xlabel = f"$\lambda\,[{_unit}]$" if self.probably_wavelength else f"$\omega\,[{_unit}]$"
        overwrite = kwargs.pop("overwrite", None)
        if overwrite is not None:
            xlabel = overwrite

        if ax is None:
            ax = self.plt
            self.plt.ylabel("I")
            self.plt.xlabel(xlabel)
            if xlim:
                self.plt.xlim(xlim)
            if ylim:
                self.plt.ylim(ylim)
            if title:
                self.plt.title(title)
        else:
            ax.set(ylabel="I")
            ax.set(xlabel=xlabel)
            if xlim:
                ax.set(xlim=xlim)
            if ylim:
                ax.set(ylim=ylim)
            if title:
                ax.set(title=title)

        if np.iscomplexobj(self.y):
            ax.plot(self.x, np.abs(self.y), color=datacolor, **kwargs)
        else:
            try:
                ax.plot(self.x, self.y_norm, color=datacolor, **kwargs)
            except (ValueError, TypeError):
                ax.plot(self.x, self.y, color=datacolor, **kwargs)
        if not nospp:
            self._plot_SPP_if_valid(ax=ax, color="black", marker="o", markersize=10, label="SPP")

    def show(self):
        """
        Equivalent with plt.show().
        """
        self.plt.show(block=True)

    @inplacify
    def normalize(self, filename=None, smoothing_level=0):
        """
        Normalize the interferogram by finding upper and lower envelope
        on an interactive matplotlib editor. Points can be deleted with
        key `d` and inserted with key `i`. Also points can be dragged
        using the mouse. On complete just close the window. Must be
        called with interactive backend. The best practice is to call
        this function inside `~pysprint.interactive` context manager.

        Parameters
        ----------
        filename : str, optional
            Save the normalized interferogram named by filename in the
            working directory. If not given it will not be saved.
            Default None.
        smoothing_level : int, optional
            The smoothing level used on the dataset before finding the
            envelopes. It applies Savitzky-Golay filter under the hood.
            Default is 0.
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
        self.plt.title("Final")
        self.plot()
        self.show()
        if filename:
            if not filename.endswith(".txt"):
                filename += ".txt"
            np.savetxt(filename, np.column_stack((self.x, self.y)), delimiter=",")
            print(f"Successfully saved as {filename}.")
        return self

    def open_SPP_panel(self, header=None):
        """
        Opens the interactive matplotlib editor for SPP data.
        Use `i` button to add a new point, use `d` key to delete one.
        The delay field is parsed to only get the numeric values.
        Close the window on finish. Must be called with interactive
        backend. The best practice is to call this function inside
        `~pysprint.interactive` context manager.

        Parameters
        ----------
        header : str, optional
            An arbitary string to include as header. This can be
            any attribute's name, or even metadata key.
        """
        if header is not None:
            if isinstance(header, str):
                head = getattr(self, header, None)
                metahead = self.meta.get(header, None)
                info = head or metahead or header
            else:
                info = None
        else:
            info = None
        spp_x, spp_y = self._prepare_SPP_data()
        _spp = SPPEditor(
            self.x, self.y_norm, info=info, x_pos=np.array(spp_x), y_pos=np.array(spp_y)
        )

        textbox = _spp._get_textbox()
        textbox.set_val(self._format_delay())

        _spp._show()

        # We need to split this into separate lines,
        # because empty results are broadcasted twice.
        delay, positions = _spp.get_data()
        self.delay, self.positions = delay, positions

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
        # Important: Use underscored variables to avoid invoking the
        # setter again, which might result in RecursionError and crashes
        # the interpreter.
        if not isinstance(self._positions, np.ndarray):
            self._positions = np.asarray(self.positions)
        if not isinstance(self.delay, np.ndarray):
            self._delay = np.ones_like(self._positions) * self._delay
        return np.atleast_1d(self.delay), np.atleast_1d(self.positions)

    def set_SPP_data(self, delay, positions, force=False):
        """
        Set the SPP data (delay and SPP positions) for the dataset.

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

        Note
        ----
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
        # trigger the callback here too
        try:
            self.callback(self, self.parent)
        except ValueError:
            pass


class MimickedDataset(Dataset):
    '''
    Class that pretends to be a dataset, but its x-y values are missing.
    It allows to set delay and SPP positions arbitrarily.
    '''
    def __init__(self, delay, positions, *args, **kwargs):
        if delay is None or positions is None:
            raise ValueError("must specify SPP data.")

        x = np.empty(1)
        y = np.empty(1)

        super().__init__(x=x, y=y, *args, **kwargs)

        self.set_SPP_data(delay=delay, positions=positions, force=True)

    @contextmanager
    def _suppress_callbacks(self):
        try:
            self.restore_parent = self.parent
            self.restore_callback = self.callback
            self.parent = None
            self.callback = lambda x, y: (_ for _ in ()).throw(ValueError('mimicked'))
            yield
        finally:
            self.parent, self.callback = self.restore_parent, self.restore_callback

    def plot(self, *args, **kwargs):
        if self.x.size == 1:
            self.plt.text(0.31, 0.5, 'Dataset is missing.', size=15)
  
    # Redefine getter-setter without boundscheck
    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, Iterable):
            for val in value:
                if not isinstance(val, numbers.Number):
                    raise ValueError(
                        f"Expected numeric values, got {type(val)} instead."
                    )

        self._positions = value
        try:
            self.callback(self, self.parent)
        except ValueError:
            pass  # delay or position is missing

    def to_dataset(self, x, y=None, ref=None, sam=None, parse=True, **kwargs):
        if parse:
            if y is not None:
                raise ValueError("cannot specify `y` explicitly if `parse=True`.")
            self.parent.ifg_names.append(x)
            if ref is not None and sam is not None:
                self.parent.sam_names.append(sam)
                self.parent.ref_names.append(ref)

            ds = Dataset.parse_raw(x, ref=ref, sam=sam, callback=self.callback, parent=self.parent, **kwargs)

        else:
            ds = Dataset(x=x, y=y, ref=ref, sam=sam, callback=self.callback, parent=self.parent, **kwargs)

        # replace the MimickedDataset with the real one
        # FIXME: need to invalidate 1 item in cache, not all

        idx = self.parent._mimicked_index(self)
        ds.parent._mimicked_set[idx] = ds
        ds.parent.__getitem__.cache_clear()

        # drop the reference
        self.parent._container.pop(self, None)
        self.parent = None
        self.callback = lambda x, y: (_ for _ in ()).throw(ValueError('mimicked'))

        # with self._suppress_callbacks():
        ds.set_SPP_data(delay=self.delay, positions=self.positions, force=True)
        return ds
