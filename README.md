# Interferometry [2019]


Interferometry is a UI for interferogram evaluation. Under construction.
I will add an advanced description later on. 

# Lastest upgrades:
  - Eval. methods are improved
  - Windows remember their last state, they open as they were closed
  - Added Settings panel for calibration
  - Added SPP Panel

### Known issues
* SPP Panel data storage should be rewritten
* Evaluation methods still performing bad
* The data loading AI sometimes produces unexpected results, currently being reviewed.
* There might be unnecessary imports
* Some buttons has no effect yet.


### To-do list

* ERROR HANDLING!
* Selectable units
* Possible performance enhancement by improving algorithms
* Possible new data manipulating features + new options for existing ones


### Installation

Interferometry requires [Python 3](https://www.python.org/downloads/) to run.

Please install the following packages:
* [PyQt5](https://pypi.org/project/PyQt5/)
* numpy, scipy, matplotlib, pandas
* [lmfit](https://lmfit.github.io/lmfit-py/), numdifftools

With command line:
```sh
$ pip install PyQt5
$ pip install numpy
$ pip install scipy
$ pip install pandas
$ pip install matplotlib
$ pip install lmfit
$ pip install numdifftools
```

With conda run:

```sh
conda install -c dsdale24 pyqt5

conda install -c conda-forge lmfit
or conda install -c conda-forge/label/gcc7 lmfit
or conda install -c conda-forge/label/broken lmfit
or conda install -c conda-forge/label/cf201901 lmfit 
```

## To Run
Run main.py
