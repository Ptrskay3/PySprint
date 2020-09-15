import warnings
from contextlib import contextmanager

import matplotlib  # noqa
import matplotlib.pyplot as plt  # noqa

from pysprint.utils.misc import run_from_ipython

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

__version__ = "0.12.5"
__author__ = "Leéh Péter"

from .core import *
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
            "change to an appropriate GUI backend. (Matplotlib 3.3.1 "
            "is broken. In that case use backend='TkAgg')."
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
            "change to an appropriate GUI backend. (Matplotlib 3.3.1"
            "is broken. In that case use backend='TkAgg')."
        ) from err
