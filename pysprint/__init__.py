import os
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in true_divide"
)


def run_from_notebook():
    """
    Detect explicitly if code is run inside Jupyter.
    """
    try:
        __IPYTHON__
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

    ipython.magic("matplotlib qt")

    import matplotlib

    plt.switch_backend("Qt5Agg")
    if matplotlib.get_backend() != "Qt5Agg":
        matplotlib.use("Qt5Agg")


if run_from_notebook():
    from matplotlib import pyplot as plt

    plt.rcParams["figure.figsize"] = [15, 5]
    try:
        from IPython import get_ipython

        ipython = get_ipython()
    except ImportError:
        import IPython.ipapi

        ipython = IPython.ipapi.get()

    ipython.magic("matplotlib qt")

    # import matplotlib
    # plt.switch_backend('Qt5Agg')
    # if matplotlib.get_backend() != 'Qt5Agg':
    #     matplotlib.use('Qt5Agg')


__version__ = "0.11.0"
__author__ = "Leéh Péter"

from .api import *
