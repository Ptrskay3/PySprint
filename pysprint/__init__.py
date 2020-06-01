import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


from pysprint.utils import run_from_ipython

# setting up the IPython notebook

if run_from_ipython():
    from matplotlib import pyplot as plt
    plt.rcParams['figure.figsize'] = [15, 5]
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        import IPython.ipapi
        ipython = IPython.ipapi.get()

    ipython.magic("matplotlib qt")

    import matplotlib
    if matplotlib.get_backend() != 'Qt5Agg':
        matplotlib.use('Qt5Agg')

__version__ = '0.10.0'
__author__ = 'Leéh Péter'

from .api import *




