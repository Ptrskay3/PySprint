import os
import warnings

import matplotlib
import matplotlib.pyplot

matplotlib.pyplot.ion()

warnings.filterwarnings(
    "ignore", message="invalid value encountered in sqrt"
)
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in true_divide"
)


def run_from_notebook():
    """
    Detect explicitly if code is run inside Jupyter.
    """
    try:
        __IPYTHON__
        # we must distinguish SPYDER because it automatically sets
        # up a backend for the user.
        if any("SPYDER" in name for name in os.environ):
            return False
        return True
    except NameError:
        return False


# setting up the IPython notebook


def setup_notebook(figsize=(15, 5), backend="Qt5Agg"):
    from matplotlib import pyplot as plt

    plt.rcParams["figure.figsize"] = figsize
    try:
        from IPython import get_ipython

        ipython = get_ipython()
    except ImportError:
        import IPython.ipapi

        ipython = IPython.ipapi.get()
    try:
        ipython.magic("matplotlib qt")
    except:
        pass

    try:
        plt.switch_backend("Qt5Agg")
        if matplotlib.get_backend() != "Qt5Agg":
            matplotlib.use("Qt5Agg")
    except:
        warnings.warn(
            "You should manually set a suitable matplotlib backend, "
            "e.g. matplotlib.use('Qt5Agg') to enable interactive plots."
        )


if run_from_notebook():
    setup_notebook()


__version__ = "0.11.0"
__author__ = "Leéh Péter"

from .api import *
