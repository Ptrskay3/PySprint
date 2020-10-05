import sys
from pathlib import Path
from typing import IO, AnyStr, Union
from textwrap import dedent
import datetime

import numpy as np

from pysprint.core.phase import Phase
from pysprint.config import _get_config_value
from pysprint.utils.exceptions import NotCalculatedException
from pysprint.utils import run_from_ipython

__all__ = ["defaultcallback", "eager_executor"]


PathOrBuffer = Union[str, Path, IO[AnyStr]]


class LogWriter:
    LABELS = ("GD", "GDD", "TOD", "FOD", "QOD", "SOD")

    def __init__(self, file: PathOrBuffer, phase: Phase, verbosity: Union[int, None] = None):
        self.file = file

        if not self.file.endswith((".log", ".txt")):
            self.file += ".log"

        # if os.path.exists(self.file):
        #     warnings.warn(f"File {self.file} exists, opening it in append mode.", PySprintWarning)

        self.phase = phase
        self.verbosity = verbosity or _get_config_value("verbosity")

    def _write_content(self, content):
        with open(self.file, "a") as logfile:
            logfile.write(content)

    def _prepare_content(self):

        if self.phase.coef_array is None:
            raise NotCalculatedException

        precision = _get_config_value("precision")

        iter_num = len(self.phase.x)

        output = dedent(f"""
        ---------------------------------------------------------------------------------------
        Date: {datetime.datetime.now()}

        Datapoints used: {iter_num}

        R^2: {self.phase._get_r_squared():.{precision}f}

        Results:
        """)

        for i, (label, value) in enumerate(zip(self.LABELS, self.phase.coef_array)):
            if value is not None and value != 0:
                output += f"{label} = {value:.{precision}f} fs^{i + 1}\n"

        if self.verbosity > 0:
            with np.printoptions(
                    threshold=sys.maxsize,
                    linewidth=np.inf,
                    precision=precision,
            ):
                output += dedent(f"""
                Values used:
                x: {self.phase.x}

                y: {self.phase.y}
                """)
        return output

    def write(self):
        content = self._prepare_content()
        self._write_content(content)


def defaultcallback(broadcaster, listener=None):
    """
    The default recorder for SPP data.
    """
    if listener is not None:
        listener._container[broadcaster] = broadcaster.emit()


def eager_executor(reference_point=None, order=None, logfile=None, verbosity=None, cleanup=True, **kwds):
    """
    This function can be used as a `callback` function. It will
    constantly monitor for SPP data changes and will re-calculate
    the dispersion on any modification.

    Parameters
    ----------
    reference_point : float
        The reference point to use.
    order : int
        The dispersion order to look for.
    cleanup : bool, optional
        Whether to flush the output and override previous results.
    logfile : string or None, optional
        If given, a logfile will be created with each iteration
    """
    if not reference_point or not order:
        raise RuntimeError(
            "Reference point and order is required."
        )

    def inner(broadcaster, listener=None):
        defaultcallback(broadcaster, listener)
        if run_from_ipython():
            from IPython.display import clear_output  # noqa
        else:
            import sys
            clear_output = sys.stdout.flush
        try:
            if cleanup:
                clear_output()
            listener.calculate(reference_point, order, **kwds)
            try:
                if logfile:
                    LogWriter(logfile, listener.GD, verbosity).write()
            except NotCalculatedException:
                pass
        except (TypeError, ValueError):
            pass

    return inner
