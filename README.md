# PySprint
### Spectrally Refined Interferometry for Python 

[![Build Status](https://travis-ci.org/Ptrskay3/PySprint.svg?branch=master)](https://travis-ci.org/Ptrskay3/pysprint)
[![Build Status](https://dev.azure.com/leehpeter/PySprint/_apis/build/status/Ptrskay3.PySprint?branchName=master)](https://dev.azure.com/leehpeter/PySprint/_build/latest?definitionId=3&branchName=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/4e876c4899af3c4435b0/maintainability)](https://codeclimate.com/github/Ptrskay3/PySprint/maintainability)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Ptrskay3/PySprint.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Ptrskay3/PySprint/context:python)
[![codecov](https://codecov.io/gh/Ptrskay3/PySprint/branch/master/graph/badge.svg)](https://codecov.io/gh/Ptrskay3/PySprint)


| | |
|-|-|
|__Name__| PySprint|
|__PyPI package__| [![PyPI version](https://badge.fury.io/py/pysprint.svg)](https://badge.fury.io/py/pysprint) |
|__Anaconda package__| [![Anaconda-Server Badge](https://anaconda.org/ptrskay/pysprint/badges/version.svg)](https://anaconda.org/ptrskay/pysprint) |
|__Development status__ | Pre-Alpha |
|__License__| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |


## Description & Features
PySprint provides an interface for Spectrally resolved interferometry in Python.


* ✓ Minimum-maximum method
* ✓ Phase modulated cosine function fit method
* ✓ Fourier method
* ✓ Stationary phase point method

### There's many more to work on..
* Windowed Fourier transform method
* Automated SPP detection, which is correctable by the user
* Better SPP interface: make the constructor accept `~pysprint.Dataset` objects.
* Calibration
* Improve file parsing
* Clean up utils, evaluate methods, edit methods
* Docstrings


Currently all the features are under testing.


### Installation

PySprint requires at least [Python 3.6](https://www.python.org/downloads/) to run.

```sh
$ pip install pysprint
```

Install on Anaconda with:
```sh
conda install -c ptrskay pysprint -c conda-forge
```

Requirements: 
* numpy
* scipy
* matplotlib 
* pandas

Optional packages: 
* lmfit - for detailed curve fitting results
* numba - to speed up non uniform FFT calculation


## Documentation

In progress.


