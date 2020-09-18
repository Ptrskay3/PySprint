import os
import warnings
from functools import lru_cache
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from pysprint.core.bases.dataset import Dataset
from pysprint.core.bases._dataset_base import _DatasetBase
from pysprint.core._evaluate import spp_method
from pysprint.utils.exceptions import DatasetError, PySprintWarning

__all__ = ["SPPMethod"]

# to suppress MatplotlibDeprecationWarning: Toggling axes navigation from
# the keyboard is deprecated since 3.3 and will be removed two minor releases later.
warnings.filterwarnings("ignore", category=cbook.mplDeprecation)


def defaultcallback(broadcaster, listener=None):
    if listener is not None:
        listener.listen(*broadcaster.emit())


class SPPMethod(metaclass=_DatasetBase):
    """
    Interface for Stationary Phase Point Method.
    """

    # TODO: kwargs should accept all the new params
    def __init__(self, ifg_names, sam_names=None, ref_names=None, errors="raise", **kwargs):
        """
        SPPMethod constructor.

        Parameters
        ----------
        ifg_names : list
            The list containing the filenames of the interferograms.
        sam_names : list, optional
            The list containing the filenames of the sample arm's spectra.
        ref_names : list, optinal
            The list containing the filenames of the reference arm's spectra.
        kwargs :
            Additional keyword arguments to pass to `parse_raw` function.
        """
        if errors not in ("raise", "ignore"):
            raise ValueError("errors must be `raise` or `ignore`.")

        self.ifg_names = ifg_names

        if sam_names:
            self.sam_names = sam_names
        else:
            self.sam_names = None
        if ref_names:
            self.ref_names = ref_names
        else:
            self.ref_names = None

        if errors == "raise":
            self._validate()
            if self.sam_names:
                if not len(self.ifg_names) == len(self.sam_names):
                    raise DatasetError(
                        "Missmatching length of files. Use None if a file is missing."
                    )
            if self.ref_names:
                if not len(self.ifg_names) == len(self.ref_names):
                    raise DatasetError(
                        "Missmatching length of files. Use None if a file is missing."
                    )

        self.skiprows = kwargs.pop("skiprows", 0)
        self.decimal = kwargs.pop("decimal", ",")
        self.sep = kwargs.pop("sep", ";")
        self.meta_len = kwargs.pop("meta_len", 1)
        self.cb = kwargs.pop("callback", defaultcallback)
        self.delimiter = kwargs.pop("delimiter", None)
        self.comment = kwargs.pop("comment", None)
        self.usecols = kwargs.pop("usecols", None)
        self.names = kwargs.pop("names", None)
        self.swapaxes = kwargs.pop("swapaxes", False)
        self.na_values = kwargs.pop("na_values", None)
        self.skip_blank_lines = kwargs.pop("skip_blank_lines", True)
        self.keep_default_na = kwargs.pop("keep_default_na", False)

        if kwargs:
            raise TypeError(f"invalid keyword argument:{kwargs}")

        self.load_dict = {
            "skiprows": self.skiprows,
            "decimal": self.decimal,
            "sep": self.sep,
            "meta_len": self.meta_len,
            "callback": self.cb,
            "delimiter": self.delimiter,
            "comment": self.comment,
            "usecols": self.usecols,
            "names": self.names,
            "swapaxes": self.swapaxes,
            "na_values": self.na_values,
            "skip_blank_lines": self.skip_blank_lines,
            "keep_default_na": self.keep_default_na
        }

        self._delay = {}
        self._positions = {}
        self._info = f"Progress: {len(self._delay)}/{len(self)}"

    def append(self, newifg, newsam=None, newref=None):
        """
        Append a new interferogram to the object.
        """
        # ensure padding before trying to append, and also
        # we better prevent infinite loop
        self.ifg_names.append(newifg)
        if newsam is not None:
            if self.sam_names is not None:
                if len(self.ifg_names) > len(self.sam_names):
                    while len(self.ifg_names) != len(self.sam_names):
                        self.sam_names.append(None)
                self.sam_names.append(newsam)
        if newref is not None:
            if self.ref_names is not None:
                if len(self.ifg_names) > len(self.ref_names):
                    while len(self.ifg_names) != len(self.ref_names):
                        self.ref_names.append(None)
                self.ref_names.append(newref)

    @staticmethod
    def calculate_from_ifg(ifg_list, reference_point, order, show_graph=False):
        """
        Collect SPP data from a list of `pysprint.Dataset` or child objects
        and evaluate them.

        Parameters
        ----------
        ifg_list : list
            The list containing the interferograms. All member should be
            `pysprint.Dataset` or child class type, otherwise TypeError is raised.
        reference_point : float
            The reference point on the x axis.
        order : int
            Maximum dispersion order to look for. Must be in [2, 6].
        show_graph : bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.

        Returns
        -------
        dispersion : array-like
            The dispersion coefficients in the form of:
            [GD, GDD, TOD, FOD, QOD, SOD]

        dispersion_std : array-like
            Standard deviations due to uncertainty of the fit.
            It is only calculated if lmfit is installed. The form is:
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std, SOD_std]

        fit_report : str
            If lmfit is available returns the fit report, else returns an
            empty string.
        """
        for ifg in ifg_list:
            if not isinstance(ifg, Dataset):
                raise TypeError("pysprint.Dataset objects are expected.")
        if order == 1:
            raise ValueError(
                "Order should be greater than 1. Cannot fit constant function to data."
            )

        local_delays = {}
        local_positions = {}

        for idx, ifg in enumerate(ifg_list):
            delay, position = ifg.emit()
            if idx != 0 and delay.flat[0] in np.concatenate(
                    [a.ravel() for a in local_delays.values()]
            ):
                raise ValueError(
                    f"Duplicated delay values found. Delay {delay.flat[0]} fs was previously seen."
                )
            local_delays[idx] = delay
            local_positions[idx] = position
        delays = np.concatenate([a.ravel() for a in local_delays.values()])
        positions = np.concatenate([a.ravel() for a in local_positions.values()])
        x, y, dispersion, dispersion_std, bf = spp_method(
            delays, positions, ref_point=reference_point, fit_order=order - 1
        )
        if show_graph:
            plt.plot(x, y, "o")
            try:
                plt.plot(x, bf, "r--", zorder=1)
            except Exception as e:
                print(e)
            plt.grid()
            plt.show(block=True)

        return dispersion, dispersion_std, bf

    def __len__(self):
        return len(self.ifg_names)

    def __str__(self):
        return f"{type(self).__name__}\nInterferogram count : {len(self)}"

    # Maybe we don't even need this.. __getitem__ seems enough
    def __iter__(self):
        try:
            for i, j, k in zip_longest(
                    self.ifg_names, self.sam_names, self.ref_names
            ):
                d = Dataset.parse_raw(
                    i,
                    j,
                    k,
                    **self.load_dict,
                    parent=self
                )
                yield d
        except TypeError:
            for i in self.ifg_names:
                d = Dataset.parse_raw(
                    i,
                    **self.load_dict,
                    parent=self
                )
                yield d

    @lru_cache()
    def __getitem__(self, key):
        try:
            dataframe = Dataset.parse_raw(
                self.ifg_names[key],
                self.sam_names[key],
                self.ref_names[key],
                **self.load_dict,
                parent=None  # a single indexing should not affect this object
            )
        except (TypeError, ValueError):
            dataframe = Dataset.parse_raw(
                self.ifg_names[key],
                **self.load_dict,
                parent=None
            )
        return dataframe

    def _validate(self):
        for filename in self.ifg_names:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"""File named '{filename}' is not found.""")
        if self.sam_names:
            for sam in self.sam_names:
                if not os.path.exists(sam):
                    raise FileNotFoundError(f"""File named '{sam}' is not found.""")
        if self.ref_names:
            for ref in self.ref_names:
                if not os.path.exists(ref):
                    raise FileNotFoundError(f"""File named '{ref}' is not found.""")

    def listen(self, delay, position):
        """
        Function which records SPP data when received.
        """
        currlen = len(self._delay)
        # TODO : Empty results are broadcasted twice..
        # It doesn't make any difference, but we'd still
        # better correct it.

        # if currlen + 1 > len(self.ifg_names):
        #     warnings.warn(
        #         "Overwriting previous values, use `reset_records` to flush the last used values.",
        #         PySprintWarning
        #     )
        self._delay[currlen] = delay
        self._positions[currlen] = position

    def reset_records(self):
        """
        Reset the state of recorded delays and positions.
        """
        self._delay = {}
        self._positions = {}

    def save_data(self, filename):
        """
        Save the currectly stored SPP data.

        Parameters
        ----------
        filename : str
            The filename to save as. If not ends with ".txt" it's
            appended by default.
        """
        if not filename.endswith(".txt"):
            filename += ".txt"
        delay = np.concatenate([_ for _ in self._delay.values()]).ravel()
        position = np.concatenate([_ for _ in self._positions.values()]).ravel()
        np.savetxt(
            f"{filename}", np.transpose(np.array([position, delay])), delimiter=",",
        )

    @staticmethod
    def calculate_from_raw(omegas, delays, reference_point, order):
        """
        Calculate the dispersion from matching pairs of delays and SPP positions.

        Parameters
        ----------
        omegas : np.ndarray
            The SPP positions.
        delays : np.ndarray
            The delay values in fs.
        reference_point : float
            The reference point on the x axis.
        order : int
            Maximum dispersion order to look for. Must be in [2, 6].

        Returns
        -------
        dispersion : array-like
            The dispersion coefficients in the form of:
            [GD, GDD, TOD, FOD, QOD, SOD]
        dispersion_std : array-like
            Standard deviations due to uncertainty of the fit.
            It is only calculated if lmfit is installed. The form is:
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std, SOD_std]
        fit_report : str
            If lmfit is available returns the fit report, else returns an
            empty string.
        """
        if order == 1:
            raise ValueError(
                "Order should be greater than 1. Cannot fit constant function to data."
            )
        x, y, dispersion, dispersion_std, bf = spp_method(
            delays, omegas, ref_point=reference_point, fit_order=order - 1
        )
        return dispersion, dispersion_std, ""

    def calculate(self, reference_point, order=2, show_graph=False):
        """
        This function should be used after setting the SPP data in
        the interactive matplotlib editor or other way.

        Parameters
        ----------
        reference_point : float
            The reference point on the x axis.
        order : int, optional
            Maximum dispersion order to look for. Must be in [2, 6].
            Default is 2.
        show_graph : bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.

        Returns
        -------
        dispersion : array-like
            The dispersion coefficients in the form of:
            [GD, GDD, TOD, FOD, QOD, SOD]
        dispersion_std : array-like
            Standard deviations due to uncertainty of the fit.
            It is only calculated if lmfit is installed. The form is:
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std, SOD_std]
        fit_report : str
            If lmfit is available returns the fit report, else returns an
            empty string.
        """
        if order == 1:
            raise ValueError(
                "Order should be greater than 1. Cannot fit constant function to data."
            )
        delays = np.concatenate([_ for _ in self._delay.values()]).ravel()
        positions = np.concatenate([_ for _ in self._positions.values()]).ravel()

        x, y, dispersion, dispersion_std, bf = spp_method(
            delays, positions, ref_point=reference_point, fit_order=order - 1
        )
        if show_graph:
            plt.plot(x, y, "o")
            try:
                plt.plot(x, bf, "r--", zorder=1)
            except Exception as e:
                print(e)
            plt.grid()

            plt.show(block=True)

        return dispersion, dispersion_std, ""

    @property
    def info(self):
        """
        Return how many interferograms where processed.
        """
        self._info = f"Progress: {len(self._delay)}/{len(self)}"
        return self._info

    @property
    def stats(self):
        raise NotImplementedError
