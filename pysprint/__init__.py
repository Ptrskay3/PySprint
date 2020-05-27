import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


from pysprint.utils import run_from_ipython
# setting up the IPython notebook
if run_from_ipython():
	from matplotlib import pyplot as plt
	plt.rcParams['figure.figsize'] = [15, 5]

__version__ = '0.6.0'
__author__ = 'Leéh Péter'

from .api import *




