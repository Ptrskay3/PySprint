
# PySprint 
[![Build Status](https://travis-ci.org/Ptrskay3/PySprint.svg?branch=master)](https://travis-ci.org/Ptrskay3/pysprint)
[![Maintainability](https://api.codeclimate.com/v1/badges/4e876c4899af3c4435b0/maintainability)](https://codeclimate.com/github/Ptrskay3/PySprint/maintainability)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Ptrskay3/PySprint.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Ptrskay3/PySprint/context:python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/Ptrskay3/PySprint/branch/master/graph/badge.svg)](https://codecov.io/gh/Ptrskay3/PySprint)



##### Spectrally Refined Interferometry for Python 
CURRENT RELEASE: 

[![PyPI version](https://badge.fury.io/py/pysprint.svg)](https://badge.fury.io/py/pysprint)

**Under construction.**

PySprint is a UI and API for interferogram evaluation. 
The GUI works only with 1920x1080 or bigger screen resolution. Lower resolution compatibility is queued on the To-Do List.

### 0.0.30 release notes and bugfixes
  - FFTMethod should work now properly, will be improved
  - Generator now works properly


### Known issues
* There can be bigger bugs out there, I'm working on a fix.
* There might be unnecessary imports
* Some buttons has no effect yet.
* In the menus some functions are unusually placed.


### To-do list
* Screen resolution compatibility
* Clean up GUI from useless buttons
* ERROR HANDLING!
* Unittests
* Selectable units
* Possible performance enhancement by improving algorithms
* Possible new data manipulating features + new options for existing ones


### Installation

PySprint requires [Python 3](https://www.python.org/downloads/) to run.

```sh
$ pip install pysprint
```

Package requirements:
* To use the GUI, install [PyQt5](https://pypi.org/project/PyQt5/)
```sh
$ pip install PyQt5
```
or 
```sh
conda install -c dsdale24 pyqt5
```
* numpy, scipy, matplotlib, pandas, lmfit will be automatically collected.


## To Run the GUI
```python
import pysprint as ps

ps.run()
```
