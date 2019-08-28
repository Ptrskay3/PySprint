# Interferometry


Interferometry is a simple GUI for interferogram evaluation. Heavily under construction.
Advanced description will be added soon.

# Lastest upgrades:
  - Added SPP Panel
  - Added some tooltips + Min-max method has new options now.
  - Added IFFT, calcuating phase comes next
  - Min-max method now uses a better fitting algorithm, fit report can be displayed
  - Peaks now send values to min-max method
  - Added Tools --> Generator, currently working on its bugs
  - New program structure
  - Added automated detection for load data.


### Known issues
* SPP Panel is under testing
* The log dialog produces bugs
* Evaluation methods can produce wrong results, currently developing them
* The data loading AI sometimes produces unexpected results, currently being reviewed.
* There might be unnecessary imports
* Generator core is under development, inconsistencies might occur.
* Some buttons has no effect yet.


### To-do list

These things will be implemented:
* ERROR HANDLING!
* Fourier Transform Method (FFT) 
* Selectable units both for angular frequency and wavelength
* Possible performance enhancement by improving algorithms
* Possible new data manipulating features


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
conda install -c conda-forge/label/gcc7 lmfit
conda install -c conda-forge/label/broken lmfit
conda install -c conda-forge/label/cf201901 lmfit 
```

## To Run
Run main.py
