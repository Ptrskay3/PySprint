"""
This file is not finished by any means.
"""
from math import factorial

import numpy as np
import matplotlib.pyplot as plt


class Phase:
    """
    A class that represents a phase obtained from various
    methods.
    """

    is_dispersion_array = False
    is_coeff = False

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.poly = None

    def __call__(self, value):
        if self.poly:
            return self.poly.__call__(value)
        raise NotImplementedError

    @classmethod
    def from_disperion_array(cls, dispersion_array, x_range=(2, 4)):
        cls.is_dispersion_array = True
        x = np.linspace(*x_range, num=1000)
        coeffs = [i * factorial(i) for i in dispersion_array]  # nem jó
        cls.poly = np.poly1d(coeffs[::-1])
        return cls(x, cls.poly(x))

    @classmethod
    def from_coeff(cls, GD, GDD=0, TOD=0, FOD=0, QOD=0, SOD=0, x_range=(2, 4)):
        x = np.linspace(*x_range, num=1000)
        cls.is_coeff = True
        cls.poly = np.poly1d([SOD, QOD, FOD, TOD, GDD, GD])
        return cls(x, cls.poly(x))

    def __str__(self):
        if self.poly is not None:
            return self.poly.__str__()

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt
        if not self.is_dispersion_array or not self.is_coeff:
            ax.plot(self.x, self.y, **kwargs)
        else:
            ax.plot(self.x, self.poly(self.x), **kwargs)

    def fit(self, order, reference_point):
        if self.is_coeff or self.is_dispersion_array:
            pass

    @property
    def data(self):
        return self.x, self.y
