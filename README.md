# PySprint

### Spectrally Resolved Interferometry for Python

![PySprint](https://github.com/ptrskay3/pysprint/actions/workflows/test.yml/badge.svg)
[![Maintainability](https://api.codeclimate.com/v1/badges/4e876c4899af3c4435b0/maintainability)](https://codeclimate.com/github/Ptrskay3/PySprint/maintainability)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Ptrskay3/PySprint.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Ptrskay3/PySprint/context:python)
[![codecov](https://codecov.io/gh/Ptrskay3/PySprint/branch/master/graph/badge.svg)](https://codecov.io/gh/Ptrskay3/PySprint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/pysprint/badge/?version=latest)](https://pysprint.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Ptrskay3/PySprint/master?filepath=index.ipynb)

| Name                      | PySprint                                                                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI package              | [![PyPI version](https://badge.fury.io/py/pysprint.svg)](https://badge.fury.io/py/pysprint)                                                                               |
| Anaconda package          | [![Not Maintained](https://img.shields.io/badge/Maintenance%20Level-Not%20Maintained-yellow.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d) |
| Development status        | Beta                                                                                                                                                                      |
| License                   | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)                                                               |
| Languages                 | Python, Rust                                                                                                                                                              |
| Supported Python versions | [![PyPI pyversions](https://img.shields.io/pypi/pyversions/pysprint.svg)](https://pypi.python.org/pypi/pysprint/)                                                         |

## Description

PySprint provides an interface for Spectrally resolved interferometry in Python.
PySprint implements all the evaluation methods described in the literature, however
the API and the software itself might change over time. Documentation is sparse and due
to the narrow use case of the software its written in Hungarian, however it will be
translated to English in the future. The testing is in _very early stages_.

- Minima-maxima method
- Phase modulated cosine function fit method
- Fourier method
- Windowed Fourier transform method
- Stationary phase point method

### Installation

PySprint requires at least [Python 3.6](https://www.python.org/downloads/) to run.

Install with

```sh
pip install pysprint
```

Requirements:

- numpy
- scipy
- matplotlib
- pandas
- Jinja2
- scikit-learn

Optional packages:

- lmfit - for detailed curve fitting results
- numba - to speed up non uniform FFT calculation
- dask - for parallel WFT run

## Documentation

The documentation is hosted on readthedocs.io. You may try the software without installing on [binder](https://mybinder.org/v2/gh/Ptrskay3/PySprint/master?filepath=index.ipynb).
