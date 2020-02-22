import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

from .conv import convert_df
from .dataset import Dataset
from .generator import Generator
from .cosfit import CosFitMethod
from ._fft import FFTMethod
from .spp import SPPMethod
from .minmax import MinMaxMethod

from pysprint.utils import run_from_ipython
# setting up the IPython notebook
if run_from_ipython():
	plt.rcParams['figure.figsize'] = [15, 5]