import os

import numpy as np
import matplotlib.pyplot as plt

from pysprint.core.bases.dataset import Dataset
from pysprint.core.bases.dataset_base import DatasetBase
from pysprint.core.evaluate import spp_method
from pysprint.utils.exceptions import DatasetError

__all__ = ["SPPMethod"]


def defaultcallback(broadcaster, listener=None):
    if listener is not None:
        listener.listen(*broadcaster.emit())


class SPPMethod(metaclass=DatasetBase):
    """
    Interface for Stationary Phase Point Method.
    """

    def __init__(self, ifg_names, sam_names=None, ref_names=None, **kwargs):
        self.ifg_names = ifg_names
        if sam_names:
            self.sam_names = sam_names
        else:
            self.sam_names = None
        if ref_names:
            self.ref_names = ref_names
        else:
            self.ref_names = None

        self._validate()
        if self.sam_names:
            if not len(self.ifg_names) == len(self.sam_names):
                raise DatasetError("Missmatching length of files.")
        if self.ref_names:
            if not len(self.ifg_names) == len(self.ref_names):
                raise DatasetError("Missmatching length of files.")
        self.idx = 0
        self.skiprows = kwargs.pop("skiprows", 8)
        self.decimal = kwargs.pop("decimal", ",")
        self.sep = kwargs.pop("sep", ";")
        self.meta_len = kwargs.pop("meta_len", 4)
        self.cb = kwargs.pop("callback", defaultcallback)

        if kwargs:
            raise TypeError(f"invalid keyword argument:{kwargs}")

        self._delay = {}
        self._positions = {}
        self._info = f"Progress: {len(self._delay)}/{len(self)}"

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
            Maximum dispersion order to look for. Must be in [2, 5].
        show_graph : bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.

        Returns
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

    def __iter__(self):
        return self

    def __str__(self):
        return f"{type(self).__name__}\nInterferogram count : {len(self)}"

    def __next__(self):
        if self.idx < len(self):
            try:
                d = Dataset.parse_raw(
                    self.ifg_names[self.idx],
                    self.sam_names[self.idx],
                    self.ref_names[self.idx],
                    skiprows=self.skiprows,
                    decimal=self.decimal,
                    sep=self.sep,
                    meta_len=self.meta_len,
                    callback=self.cb,
                    parent=self
                )
            except TypeError:
                d = Dataset.parse_raw(
                    self.ifg_names[self.idx],
                    skiprows=self.skiprows,
                    decimal=self.decimal,
                    sep=self.sep,
                    meta_len=self.meta_len,
                    callback=self.cb,
                    parent=self
                )
            self.idx += 1
            return d
        raise StopIteration

    def __getitem__(self, key):
        try:
            dataframe = Dataset.parse_raw(
                self.ifg_names[key],
                self.sam_names[key],
                self.ref_names[key],
                skiprows=self.skiprows,
                decimal=self.decimal,
                sep=self.sep,
                meta_len=self.meta_len,
                callback=self.cb,
                parent=None  # a single indexing should not affect this object
            )
        except (TypeError, ValueError):
            dataframe = Dataset.parse_raw(
                self.ifg_names[key],
                skiprows=self.skiprows,
                decimal=self.decimal,
                sep=self.sep,
                meta_len=self.meta_len,
                callback=self.cb,
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
        self._delay[self.idx] = delay
        self._positions[self.idx] = position

    def save_data(self, filename):
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
            Maximum dispersion order to look for. Must be in [2, 5].

        Returns
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
        the interactive matplotlib editor.

        Parameters
        ----------
        reference_point : float
            The reference point on the x axis.

        order : int, optional
            Maximum dispersion order to look for. Must be in [2, 5].
            Default is 2.

        show_graph : bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.

        Returns
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
        self._info = f"Progress: {len(self._delay)}/{len(self)}"
        return self._info

    @property
    def stats(self):
        raise NotImplementedError
