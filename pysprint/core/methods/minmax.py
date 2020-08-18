import logging

import numpy as np

from pysprint.core.bases.dataset import Dataset
from pysprint.mpl_tools.peak import EditPeak
from pysprint.core.phase import Phase
from pysprint.core.evaluate import min_max_method, is_inside
from pysprint.utils import (
    _maybe_increase_before_cwt,
    calc_envelope,
)

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

    def init_edit_session(self, engine="normal", **kwargs):
        """
        Function to initialize peak editing on a plot.
        Right clicks (`d` key later) will delete the closest point,
        left clicks(`i` key later) will add a new point. Just close
        the window when finished.

        Parameters:
        ----------

        engine : str
            Must be 'cwt', 'normal' or 'slope'.
            Peak detection algorithm to use.
            Default is normal.

        **kwargs:
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
            _x, _y, _xx, _yy = self.detect_peak(
                pmax=pmax, pmin=pmin, threshold=threshold, except_around=except_around,
            )

            # just for validation purposes
            _ = kwargs.pop("widths", np.arange(1, 20))
            _ = kwargs.pop("floor_thres", 0.05)

        elif engine == "slope":
            x, _, _, _ = self._safe_cast()
            y = np.copy(self.y_norm)
            if _maybe_increase_before_cwt(y):
                y += 2
            _, lp, lloc = calc_envelope(y, np.arange(len(y)), "l")
            _, up, uloc = calc_envelope(y, np.arange(len(y)), "u")
            lp -= 2
            up -= 2
            _x, _xx = x[lloc], x[uloc]
            _y, _yy = lp, up

        elif engine == "cwt":
            widths = kwargs.pop("widths", np.arange(1, 20))
            floor_thres = kwargs.pop("floor_thres", 0.05)
            _x, _y, _xx, _yy = self.detect_peak_cwt(
                widths=widths, floor_thres=floor_thres
            )

            # just for validation purposes
            _ = kwargs.pop("pmax", 0.1)
            _ = kwargs.pop("pmin", 0.1)
            _ = kwargs.pop("threshold", 0.1)
            _ = kwargs.pop("except_around", None)

        _xm = np.append(_x, _xx)
        _ym = np.append(_y, _yy)

        if kwargs:
            raise TypeError(f"Invalid argument:{kwargs}")

        try:
            _editpeak = EditPeak(self.x, self.y_norm, _xm, _ym)
        except ValueError:
            _editpeak = EditPeak(self.x, self.y, _xm, _ym)
        # Automatically propagate these points to the mins and maxes.
        # Better distribute these points between min and max, just in case
        # the default argrelextrema is definitely not called
        # in `pysprint.core.evaluate.min_max_method`.

        # TODO : Don't distribute them, we'd rather sort by values to find
        # out it's a min or max.
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
            allow_parallel=False
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
        show_graph: bool, optional
            Shows a the final graph of the spectral phase and fitted curve.
            Default is False.

        Returns
        -------
        dispersion: array-like
            [GD, GDD, TOD, FOD, QOD]
        dispersion_std: array-like
            Standard deviations due to uncertainty of the fit.
            They are only calculated if lmfit is installed.
            [GD_std, GDD_std, TOD_std, FOD_std, QOD_std]
        fit_report: string
            lmfit report if installed, else empty string.

        Notes:
        ------
        Decorated with print_disp, so the results are
        immediately printed without explicitly saying so.
        """

        phase = self.build_phase(reference_point=reference_point, SPP_callbacks=SPP_callbacks)

        if is_inside(reference_point, phase.x):
            left_phase = phase.slice(None, reference_point, inplace=False)
            right_phase = phase.slice(reference_point, None, inplace=False)
        else:
            allow_parallel = False
            logger.info("Parallel is disabled, reference_point is on the border.")

        if allow_parallel:
            left_d, left_ds, left_fit_report = left_phase._fit(reference_point, order)
            right_d, right_ds, right_fit_report = right_phase._fit(reference_point, order)

            logger.info(f"left side evaluated to {left_d}, used {len(left_phase.x)} points.")
            logger.info(f"right side evaluated to {right_d}, used {len(right_phase.x)} points.")

            right_d = np.where(np.sign(left_d) != np.sign(right_d), -right_d, right_d)
            right_ds = np.where(np.sign(left_ds) != np.sign(right_ds), -right_ds, right_ds)

            diffs = np.abs(np.trim_zeros(right_d) - np.trim_zeros(left_d)) / np.trim_zeros(left_d)

            if (diffs[~np.isnan(diffs)] > 0.5).any():
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

    def build_phase(self, reference_point, SPP_callbacks=None):
        if SPP_callbacks is None and self._positions is not None:
            SPP_callbacks = np.array(self._positions)

        x, y = min_max_method(
            self.x,
            self.y,
            self.ref,
            self.sam,
            ref_point=reference_point,
            maxx=self.xmax,
            minx=self.xmin,
            SPP_callbacks=SPP_callbacks
        )
        self.phase = Phase(x, y)
        return self.phase
