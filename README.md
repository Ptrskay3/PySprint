# Interferometry [2019]


Interferometry is a UI for interferogram evaluation. Under construction, error handling is non-existant for the time being.
I will add an advanced description later on. 

# Lastest upgrades:
  - Added Settings panel
  - Added SPP Panel
  - Added some tooltips + Min-max method has new options now.
  - Added IFFT
  - Min-max method now uses a better fitting algorithm, fit report can be displayed
  - Peaks now send values to min-max method
  - Added Tools --> Generator
  - New program structure
  - Added automated detection for load data.


### Known issues
* SPP Panel might have tons of issues
* The log dialog produces bugs
* Evaluation methods can produce wrong results
* The data loading AI sometimes produces unexpected results, currently being reviewed.
* There might be unnecessary imports
* Generator core is pre-alpha.
* Some buttons has no effect yet.


### To-do list

* ERROR HANDLING!
* Fourier Transform Method (FFT) 
* Selectable units both for angular frequency and wavelength
* Possible performance enhancement by improving algorithms
* Possible new data manipulating features
* Interferometer base dispersion should be added for calibration. 

### Installation

Interferogram requires [Python 3](https://www.python.org/downloads/) to run.

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
