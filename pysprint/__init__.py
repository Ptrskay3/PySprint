import warnings

import matplotlib # noqa
import matplotlib.pyplot as plt

from pysprint.utils.misc import run_from_ipython


warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


def interactive(figsize=(15, 5), backend="Qt5Agg"):
    plt.ion()
    plt.rcParams["figure.figsize"] = figsize
    try:
        from IPython import get_ipython

        ipython = get_ipython()
    except ImportError:
        import IPython.ipapi

        ipython = IPython.ipapi.get()
    try:
        ipython.magic("matplotlib qt")
    except AttributeError:
        pass
    else:
        pass

    try:
        plt.switch_backend(backend)
    except AttributeError:
        warnings.warn(
            "You should manually set a suitable matplotlib backend, "
            "e.g. matplotlib.use('Qt5Agg') to enable interactive plots."
        )


__version__ = "0.12.2"
__author__ = "Leéh Péter"

from .api import *
from .utils import print_info
