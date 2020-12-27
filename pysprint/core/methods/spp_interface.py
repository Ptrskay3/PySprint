import os
import warnings
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt  # noqa
import matplotlib.cbook as cbook

from pysprint.core.bases.dataset import Dataset
from pysprint.core.bases.dataset import MimickedDataset
from pysprint.core.bases._dataset_base import _DatasetBase
from pysprint.core.phase import Phase
from pysprint.utils.exceptions import DatasetError
from pysprint.core.callbacks import defaultcallback
from pysprint.utils.exceptions import PySprintWarning
from pysprint.utils import from_pat

__all__ = ["SPPMethod"]

warnings.simplefilter("ignore", category=cbook.MatplotlibDeprecationWarning)


class SPPMethod(metaclass=_DatasetBase):
    """
    Interface for Stationary Phase Point Method.
    """

    def __init__(self, ifg_names, sam_names=None, ref_names=None, errors="raise", implements=None, **kwargs):
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

        if implements is None:
            self.implements = Dataset
        else:
            if issubclass(implements, Dataset):
                self.implements = implements
            else:
                raise TypeError("implementer must subclass `pysprint.Dataset`.")

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

        self._container = {}
        self.GD = None
        self._mimicked = False
        self._mimicked_set = []

    @classmethod
    def from_pattern(cls, pattern, mod=None, **kwargs):
        """
        Load all files matching a pattern, then optinally sort them
        by modulo 3 if there are arms' spectra.
        
        Parameters
        ----------
        pattern : str
            The pattern to match. For example to load all txt files in the
            current directory, use `"*.txt"`.
        mod : int, optional
            The modulus to sort files by. Set this option to 3 if you have
            consecutive measurements, which contain arms' spectra. The option
            mod = -1 will group by modulo 3, but it will only include every 
            third file.
        kwargs :
            Additional keyword arguments for the original constructor, for
            example loading options.
        """
        return cls(**from_pat(pattern, mod), **kwargs)

    @classmethod
    def from_log(cls, logfile, **kwargs):
        """
        Restore a checkpoint from logfile. It creates MimickedDataset
        objects, which has no x-y values set, only contains delay and
        SPP position values. These values can be set without restrictions.

        Parameters
        ----------
        logfile : str
            The logfile created by pysprint with verbosity level 1 or greater.
        kwargs : optional
            Additional keyword arguments to pass to MimickedDataset (for example
            callback might be important).
        """
        callback = kwargs.pop("callback", defaultcallback)
        instance = cls([], callback=callback)
        _mimicked_set = []
        GD = Phase.from_log(logfile, GD_mode=True)
        xx, yy = _group_multiple_positions(GD.x, GD.y)
        for pos, delay in zip(xx, yy):
            ds = MimickedDataset(
                delay=delay[0],
                positions=pos,
                parent=instance,
                callback=callback,
                **kwargs
            )
            _mimicked_set.append(ds)
        instance._mimicked = True
        instance._mimicked_set = _mimicked_set
        return instance

    def _mimicked_index(self, member):
        for i, ds in enumerate(self):
            if ds == member:
                return i

    def _collect(self):

        # Maybe the dictionary struct can be dropped at this point..
        local_delays = {}
        local_positions = {}

        for idx, (delay, position) in enumerate(self._container.values()):
            if idx != 0 and delay.size > 0 and delay.flat[0] in np.concatenate(
                    [a.ravel() for a in local_delays.values()]
            ):
                warnings.warn(
                    f"Duplicated delay values found. Delay {delay.flat[0]} fs was previously seen.",
                    PySprintWarning
                )
            local_delays[idx] = delay
            local_positions[idx] = position

        delays = np.concatenate([a.ravel() for a in local_delays.values()])
        positions = np.concatenate([a.ravel() for a in local_positions.values()])

        return delays, positions

    def append(self, newifg, newsam=None, newref=None):
        """
        Append a new interferogram to the object.
        """
        # ensure padding before trying to append, and also
        # we better prevent infinite loop

        # TODO: This seems wrong..
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
            if idx != 0 and delay.size > 0 and delay.flat[0] in np.concatenate(
                    [a.ravel() for a in local_delays.values()]
            ):
                warnings.warn(
                    f"Duplicated delay values found. Delay {delay.flat[0]} fs was previously seen.",
                    PySprintWarning
                )
            local_delays[idx] = delay
            local_positions[idx] = position
        delays = np.concatenate([a.ravel() for a in local_delays.values()])
        positions = np.concatenate([a.ravel() for a in local_positions.values()])
        GD = Phase(positions, delays, GD_mode=True)
        d, ds, s = GD._fit(reference_point=reference_point, order=order)

        if show_graph:
            GD.plot()

        return d, ds, s

    def __len__(self):
        l = len(self.ifg_names)
        g = len(self._mimicked_set)
        return max([l, g])

    # TODO
    def __str__(self):
        return f"{type(self).__name__}\nInterferogram count : {len(self)}"

    def _repr_html_(self):
        alive = [i for i in Dataset._get_instances() if i.parent == self]
        s = f"""
        <table style="border:1px solid black;float:top;">
        <tbody>
        <tr>
        <td colspan=2 style="text-align:center">
        <font size="5">{type(self).__name__}</font>
        </td>
        </tr>
        <tr>
        <td style="text-align:center"><b>Interferograms accumulated<b></td>
            <td style="text-align:center"> {len(self)}</td>
        </tr>
        <tr>
        <td style="text-align:center"><b>Interferograms cached<b></td>
            <td style="text-align:center"> {len(alive)}</td>
        </tr>
        <tr>
        <td style="text-align:center"><b>Data recorded from<b></td>
            <td style="text-align:center"> {len(self._container)}</td>
        </tr>
        <tr>
        <td style="text-align:center"><b>Eagerly calculating<b></td>
            <td style="text-align:center"> {self.is_eager}</td>
        </tr>
        </table>
        """
        return s

    @lru_cache(500)
    def __getitem__(self, key):
        if not self._mimicked:
            if isinstance(key, slice):
                raise TypeError(
                    "Slices are not acceptable, but will be in the future versions."
                )
            try:
                dataframe = self.implements.parse_raw(
                    self.ifg_names[key],
                    self.sam_names[key],
                    self.ref_names[key],
                    **self.load_dict,
                    parent=self
                )
            except (TypeError, ValueError):
                dataframe = self.implements.parse_raw(
                    self.ifg_names[key],
                    **self.load_dict,
                    parent=self
                )
            return dataframe
        else:
            return self._mimicked_set[key]

    def _validate(self):
        for filename in self.ifg_names:
            if filename is not None and not os.path.exists(filename):
                raise FileNotFoundError(f"""File named '{filename}' is not found.""")
        if self.sam_names:
            for sam in self.sam_names:
                if sam is not None and not os.path.exists(sam):
                    raise FileNotFoundError(f"""File named '{sam}' is not found.""")
        if self.ref_names:
            for ref in self.ref_names:
                if ref is not None and not os.path.exists(ref):
                    raise FileNotFoundError(f"""File named '{ref}' is not found.""")

    def flush(self):
        """
        Reset the state of recorded delays and positions, even on
        active objects that have been constructed on the runtime.
        """
        self._container = {}

        for ifg in self:
            ifg._delay = None
            ifg.delay = None
            ifg._positions = None
            ifg.positions = None
        
        # Clean up all other instances which might have been omitted due to inplace ops.
        alive = [i for i in Dataset._get_instances() if i.parent == self]
        for ifg in alive:
            ifg._delay = None
            ifg.delay = None
            ifg._positions = None
            ifg.positions = None


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
        delays, positions = self._collect()
        np.savetxt(
            f"{filename}", np.column_stack((positions, delays)), delimiter=",",
        )

    @staticmethod
    def calculate_from_raw(omegas, delays, reference_point, order, show_graph=False):
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
        show_graph : bool, optional
            Whether to show the fitting.

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

        GD = Phase(omegas, delays, GD_mode=True)
        d, ds, s = GD._fit(reference_point=reference_point, order=order)

        if show_graph:
            GD.plot()

        return d, ds, s

    def build_GD(self):
        """
        Build the GD and return it. It will rebuild every time
        this function is called.
        """
        delays, positions = self._collect()

        self.GD = Phase(positions, delays, GD_mode=True)
        return self.GD

    def calculate(self, reference_point, order, show_graph=False):
        """
        This function should be used after setting the SPP data in
        the interactive matplotlib editor or other way.

        Parameters
        ----------
        reference_point : float
            The reference point on the x axis.
        order : int, optional
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
        if order == 1:
            raise ValueError(
                "Order should be greater than 1. Cannot fit constant function to data."
            )
        self.build_GD()
        d, ds, s = self.GD._fit(reference_point=reference_point, order=order)

        if show_graph:
            self.GD.plot()

        return d, ds, s

    @property
    def info(self):
        """
        Return how many interferograms were processed.
        """
        self._info = f"Progress: {len(self._container)}/{len(self)}"
        return self._info

    @property
    def is_eager(self):
        """
        Returns if eager execution is enabled.
        """
        # TODO
        if self.cb.__name__ == "inner":
            return True
        return False


def _group_multiple_positions(x, y):
    ys = np.split(y, np.where(np.diff(y) != 0)[0] + 1)
    xs = np.split(x, np.where(np.diff(y) != 0)[0] + 1)
    return xs, ys