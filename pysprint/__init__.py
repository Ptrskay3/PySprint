import warnings
from contextlib import contextmanager

import matplotlib  # noqa
import matplotlib.pyplot as plt  # noqa

from pysprint.utils.misc import run_from_ipython

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


@contextmanager
def interactive(backend="Qt5Agg", figsize=(15, 5)):
    """
    Context manager to temporarily change the matplotlib
    backend to ensure interactive figure are rendered
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
    except AttributeError as err:
        raise ValueError(
            f"Couldn't set backend {backend}, you should manually "
            "change to an appropriate GUI backend."
        ) from err
    finally:
        plt.switch_backend(original_backend)


def set_interactive(backend="Qt5Agg", figsize=(15, 5)):
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
    except AttributeError as err:
        raise ValueError(
            f"Couldn't set backend {backend}, you should manually "
            "change to an appropriate GUI backend."
        ) from err


__version__ = "0.12.3"
__author__ = "Leéh Péter"

from .core import *
from .utils import print_info
