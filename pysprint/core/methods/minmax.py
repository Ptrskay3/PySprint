import numpy as np

from pysprint.core.bases.dataset import Dataset
from pysprint.mpl_tools.peak import EditPeak
from pysprint.utils import (
    _maybe_increase_before_cwt,
    calc_envelope,
)
from pysprint.core.evaluate import min_max_method

__all__ = ["MinMaxMethod"]


class MinMaxMethod(Dataset):
    """
    Interface for Minimum-Maximum Method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                pmax=pmax,
                pmin=pmin,
                threshold=threshold,
                except_around=except_around,
            )

            # just for validation purposes
            _ = kwargs.pop("width", 10)
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
            width = kwargs.pop("width", 10)
            floor_thres = kwargs.pop("floor_thres", 0.05)
            _x, _y, _xx, _yy = self.detect_peak_cwt(
                width=width, floor_thres=floor_thres
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
        self.xmin = _editpeak.get_dat[0][:len(_editpeak.get_dat[0]) // 2]
        self.xmax = _editpeak.get_dat[0][len(_editpeak.get_dat[0]) // 2:]
        print(
            f"{len(_editpeak.get_dat[0])} extremal points were recorded."
        )
        return _editpeak.get_dat[0]  # we should return None

    def calculate(self, reference_point, order, show_graph=False):
        """
        MinMaxMethod's calculate function.

        Parameters
        ----------
        reference_point : float
            reference point on x axis

        order : int
            Polynomial (and maximum dispersion) order to fit. Must be in [1, 5].

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
        dispersion, dispersion_std, fit_report = min_max_method(
            self.x,
            self.y,
            self.ref,
            self.sam,
            ref_point=reference_point,
            maxx=self.xmax,
            minx=self.xmin,
            fit_order=order,
            show_graph=show_graph,
        )
        self._dispersion_array = dispersion
        return dispersion, dispersion_std, fit_report
