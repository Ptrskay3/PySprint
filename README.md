# Interferometry


Interferometry is a simple GUI for interferogram evaluation. Heavily under construction.
Advanced description will be added soon.

# Lastest upgrades:
  - Added Tools --> Generator, currently working on its bugs
  - New program structure
  - Added automated detection for load data.
  - Added Swap axes + some bugfixes
  - Added FFT ( Gaussian Window filtering and IFFT next)
  - Added Modify data panel

### Known issuesn
* Evaluation methods can produce wrong results, currently developing them
* Unit selector has no effect yet
* The data loading AI sometimes produces unexpected results, currently being reviewed.
* There might be unnecessary imports
* Generator core is under development, inconsistencies might occur.
* Some buttons has no effect yet.


### Coming soon..

These things will be implemented:
* ERROR HANDLING!
* Stationary Phase Point Method (SPP)
* Fourier Transform Method (FFT) 
* Selectable units both for angular frequency and wavelength
* Possible performance enhancement by improving algorithms
* Possible new data manipulating features


### Installation

Interferogram requires [Python 3](https://www.python.org/downloads/) to run.

Please install the following packages:
* [PyQt5](https://pypi.org/project/PyQt5/)
* numpy, scipy, matplotlib, pandas, lmfit, numdifftools

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
```

## To Run
Run main.py
