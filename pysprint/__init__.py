﻿try:
    from .pysprint import blank, set_panic_hook
    # setup a global ctrl-c handler
    set_panic_hook()
except Exception:
    def blank(*args, **kwargs):
        raise ImportError("Rust Extensions aren't built.")

import sys

import warnings
from contextlib import contextmanager

import matplotlib  # noqa
import matplotlib.pyplot as plt  # noqa

import pysprint.core.init_config
from pysprint.utils.misc import run_from_ipython


# Warnings related to FFT.. They should be ignored in 99% of cases.
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")


__author__ = "Leéh Péter"

from .core import *
import pysprint.devices
from .utils import print_info

default_backend = "Qt5Agg" if getattr(matplotlib, "__version__", None) != "3.3.1" else "TkAgg"


@contextmanager
def interactive(backend=default_backend, figsize=(15, 5)):
    """
    Context manager to temporarily change the matplotlib
    backend to ensure interactive figures are rendered
    correctly.

    Parameters
    ----------
    backend : string, optional
        The matplotlib backend to use. Default is "Qt5Agg".
    figsize : tuple, optional
        The figure size to use. Default is (15, 5).
    """
    plt.ion()
    matplotlib.rcParams["figure.figsize"] = figsize
    original_backend = plt.get_backend()
    try:
        plt.switch_backend(backend)
        yield
    except (AttributeError, ImportError, ModuleNotFoundError) as err:
        raise ValueError(
            f"Couldn't set backend {backend}, you should manually "
            "change to an appropriate GUI backend."
        ) from err
    finally:
        plt.switch_backend(original_backend)


def set_interactive(backend=default_backend, figsize=(15, 5)):
    """
    Set the backend for matplotlib permanently.

    Parameters
    ----------
    backend : string, optional
        The matplotlib backend to use. Default is "Qt5Agg".
    figsize : tuple, optional
        The figure size to use. Default is (15, 5).
    """
    plt.ion()
    matplotlib.rcParams["figure.figsize"] = figsize
    try:
        plt.switch_backend(backend)
    except (AttributeError, ImportError, ModuleNotFoundError) as err:
        raise ValueError(
            f"Couldn't set backend {backend}, you should manually "
            "change to an appropriate GUI backend."
        ) from err


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
