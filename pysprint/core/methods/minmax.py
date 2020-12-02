import logging
import warnings

import numpy as np

from pysprint.config import _get_config_value
from pysprint.core.bases.dataset import Dataset
from pysprint.mpl_tools.peak import EditPeak
from pysprint.core.phase import Phase
from pysprint.core._evaluate import min_max_method, is_inside
from pysprint.utils import (
    _maybe_increase_before_cwt,
    _calc_envelope,
)
from pysprint.utils import PySprintWarning


logger = logging.getLogger(__name__)
FORMAT = "[ %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

__all__ = ["MinMaxMethod"]


class MinMaxMethod(Dataset):
    """
    Interface for Minimum-Maximum Method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = None
        self._is_onesided = False

    def init_edit_session(self, engine="normal", **kwargs):
        """
        Function to initialize peak editing on a plot.
        Right clicks (`d` key later) will delete the closest point,
        left clicks(`i` key later) will add a new point. Just close
        the window when finished. Must be called with interactive
        backend. The best practice is to call this function inside
        `~pysprint.interactive` context manager.

        Parameters
        ----------
        engine : str, optional
            Must be 'cwt', 'normal' or 'slope'.
            Peak detection algorithm to use.
            Default is normal.
        kwargs : dict, optional
            pmax, pmin, threshold, except_around, width
        """
        engines = ("cwt", "normal", "slope")

        if engine not in engines:
            raise ValueError(f"Engine must be in {str(engines)}")

        if engine == "normal":
            pmax = kwargs.pop("pmax", 0.1)
            pmin = kwargs.pop("pmin", 0.1)
            threshold = kwargs.pop("threshold", 0.1)
            except_around = kwargs.pop("except_around", None)
            side = kwargs.pop("side", "both")
            _x, _y, _xx, _yy = self.detect_peak(
                pmax=pmax, pmin=pmin, threshold=threshold, except_around=except_around, side=side
            )

            # just for validation purposes
            _ = kwargs.pop("widths", np.arange(1, 20))
            _ = kwargs.pop("floor_thres", 0.05)

        elif engine == "slope":
            side = kwargs.pop("side", "both")
            self._is_onesided = side != "both"
            x, _, _, _ = self._safe_cast()
            y = np.copy(self.y_norm)
            if _maybe_increase_before_cwt(y):
                y += 2
            _, lp, lloc = _calc_envelope(y, np.arange(len(y)), "l")
            _, up, uloc = _calc_envelope(y, np.arange(len(y)), "u")
            lp -= 2
            up -= 2
            _x, _xx = x[lloc], x[uloc]
            _y, _yy = lp, up

        elif engine == "cwt":
            widths = kwargs.pop("widths", np.arange(1, 20))
            side = kwargs.pop("side", "both")
            floor_thres = kwargs.pop("floor_thres", 0.05)
            _x, _y, _xx, _yy = self.detect_peak_cwt(
                widths=widths, floor_thres=floor_thres, side=side
            )

            # just for validation purposes
            _ = kwargs.pop("pmax", 0.1)
            _ = kwargs.pop("pmin", 0.1)
            _ = kwargs.pop("threshold", 0.1)
            _ = kwargs.pop("except_around", None)

        if side == "both":
            _xm = np.append(_x, _xx)
            _ym = np.append(_y, _yy)
        elif side == "min":
            _xm, _ym = _xx, _yy
        elif side == "max":
            _xm, _ym = _x, _y
        else:
            raise ValueError("Side must be 'both', 'min' or 'max'.")

        if kwargs:
            raise TypeError(f"Invalid argument:{kwargs}")

        try:
            _editpeak = EditPeak(self.x, self.y_norm, _xm, _ym)
        except ValueError:
            _editpeak = EditPeak(self.x, self.y, _xm, _ym)
        # Automatically propagate these points to the mins and maxes.
        # Distribute these points between min and max, just in case
        # the default argrelextrema is definitely not called
        # in `pysprint.core.evaluate.min_max_method`.

        self.xmin = _editpeak.get_dat[0][:len(_editpeak.get_dat[0]) // 2]
        self.xmax = _editpeak.get_dat[0][len(_editpeak.get_dat[0]) // 2:]
        print(f"{len(_editpeak.get_dat[0])} extremal points were recorded.")
        return _editpeak.get_dat[0]  # we should return None

    def calculate(
            self,
            reference_point,
            order,
            SPP_callbacks=None,
            show_graph=False,
            scan=False,
            onesided=False,
    ):
        """
        MinMaxMethod's calculate function.

        Parameters
        ----------
        reference_point : float
            reference point on x axis
        order : int
            Polynomial (and maximum dispersion) order to fit. Must be in [1, 5].
        SPP_callbacks : number, or numeric list-like
            The positions of SPP's on the interferogram. If not given it will check
            if there's any SPP position set on the object.
        show_graph : bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.
        onesided : bool
            Use only minimums or maximums to build the phase. It also works for
            one characteristic point per oscillation period (e.g. zero-crossings).
            Default is False.

        Returns
        -------
        dispersion : array-like
            [GD, GDD, TOD, FOD, QOD, SOD]
        dispersion_std : array-like
            Standard deviations due to uncertainty of the fit.
            They are only calculated if lmfit is installed.
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std, SOD_std]
        fit_report : str
            lmfit report if installed, else empty string.

        Note
        ----
        Decorated with pprint_disp, so the results are
        immediately printed without explicitly saying so.
        """

        phase = self.build_phase(
            reference_point=reference_point, SPP_callbacks=SPP_callbacks, onesided=onesided
        )

        if is_inside(reference_point, phase.x):
            left_phase = phase.slice(None, reference_point, inplace=False)
            right_phase = phase.slice(reference_point, None, inplace=False)
        else:
            scan = False
            logger.info("Scan is disabled, reference_point is on the border.")

        if scan:
            left_d, left_ds, left_fit_report = left_phase._fit(reference_point, order)
            right_d, right_ds, right_fit_report = right_phase._fit(reference_point, order)

            logger.info(f"left side evaluated to {left_d}, used {len(left_phase.x)} points.")
            logger.info(f"right side evaluated to {right_d}, used {len(right_phase.x)} points.")

            right_d = np.where(np.sign(left_d) != np.sign(right_d), -right_d, right_d)
            right_ds = np.where(np.sign(left_ds) != np.sign(right_ds), -right_ds, right_ds)

            diffs = np.abs(np.trim_zeros(right_d) - np.trim_zeros(left_d)) / np.trim_zeros(left_d)

            thres = _get_config_value("scan_threshold")

            if (np.abs(diffs[~np.isnan(diffs)]) > thres).any():
                dispersion = left_d if len(left_phase.x) >= len(right_phase.x) else right_d
                dispersion_std = left_ds if len(left_phase.x) >= len(right_phase.x) else right_ds
                fit_report = left_fit_report if len(left_phase.x) >= len(right_phase.x) else right_fit_report
                side = "left" if len(left_phase.x) >= len(right_phase.x) else "right"
                logger.info(f"Max relative difference is too high, using {side} side.")

                if side == "left":
                    if show_graph:
                        phase.plot()
                        left_phase.plot(label="used part")
                else:
                    if show_graph:
                        phase.plot()
                        right_phase.plot(label="used part")

            else:
                dispersion = np.mean([left_d, right_d], axis=0)

                dispersion_std = np.mean([left_ds, right_ds], axis=0)

                fit_report = ''.join([left_fit_report, right_fit_report])
                if show_graph:
                    left_phase.plot()
                    right_phase.plot()
        else:
            dispersion, dispersion_std, fit_report = phase._fit(reference_point, order)
            if show_graph:
                phase.plot()

        return dispersion, dispersion_std, fit_report

    def build_phase(self, reference_point, SPP_callbacks=None, onesided=False):
        """
        Build **only the phase** using reference point and SPP positions.

        Parameters
        ----------
        reference_point : float
            The reference point from where the phase building starts.
        SPP_callbacks : number, or numeric list-like
            The positions of SPP's on the interferogram. If not given it will check
            if there's any SPP position set on the object.
        onesided : bool
            If `True`, use only the minimums or maximums to build the phase. It also
            works for one characteristic point per oscillation period (e.g. zero-crossings).
            Default is False.

        Returns
        -------
        phase : pysprint.core.phase.Phase
            The phase object. See its docstring for more info.
        """
        if SPP_callbacks is None and self._positions is not None:
            SPP_callbacks = np.array(self._positions)

        if onesided and not self._is_onesided:
            warnings.warn(
                "Trying to build phase as one-sided, but the detection was two-sided. Use `onesided=False`.",
                PySprintWarning
            )

        if not onesided and self._is_onesided:
            warnings.warn(
                "Trying to build phase as two-sided, but the detection was one-sided. Use `onesided=True`.",
                PySprintWarning
            )

        x, y = min_max_method(
            self.x,
            self.y,
            self.ref,
            self.sam,
            ref_point=reference_point,
            maxx=self.xmax,
            minx=self.xmin,
            SPP_callbacks=SPP_callbacks,
            onesided=onesided,
        )
        self.phase = Phase(x, y)
        return self.phase
