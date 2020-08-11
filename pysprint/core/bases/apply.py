import types
import warnings
from collections.abc import Iterable
from inspect import getfullargspec

import numpy as np


class DatasetApply:
    def __init__(
            self,
            obj,
            func,
            axis=None,
            args=None,
            kwargs=None
    ):

        self.obj = obj
        self.args = args or ()
        self.kwargs = kwargs or {}

        self.f = func
        self.axis = axis

        if self.axis == "x" or self.axis == 0:
            self.target = "x"
        elif self.axis == "y" or self.axis == 1:
            self.target = "y"
        else:
            raise ValueError("Axis must be 'x', 'y', '0' or '1'.")
        self.shape = len(getattr(self.obj, self.target))

    def perform(self):

        if isinstance(self.f, str):
            func = getattr(self.obj, self.f)
            sig = getfullargspec(func)
            if "axis" in sig.args:
                self.kwargs["axis"] = self.axis
            # Let's assume we don't mess up the shape internally
            func(*self.args, **self.kwargs)
            return self.obj  # we need to return this because of `inplacify` deco.

        elif isinstance(self.f, np.ufunc):
            target = getattr(self.obj, self.target)

            retval = self.f(target, *self.args, **self.kwargs)
            value = self._validate(retval)

            setattr(self.obj, self.target, value)
            if self.target == "y":
                setattr(self.obj, "y_norm", value)
            return value

        elif isinstance(self.f, types.FunctionType):
            sig = getfullargspec(self.f)
            if "axis" in sig.args:
                self.kwargs["axis"] = self.axis
            # we can safely vectorize it here
            self.f = np.vectorize(self.f)
            target = getattr(self.obj, self.target)
            retval = self.f(target, *self.args, **self.kwargs)
            value = self._validate(retval)
            setattr(self.obj, self.target, value)
            if self.target == "y":
                setattr(self.obj, "y_norm", value)
            return value

    def _validate(self, val):

        if isinstance(val, (Iterable, np.ndarray)):
            val = np.asarray(val, dtype=np.float64)

            if val.ndim != 1:
                val = np.concatenate(val).ravel()
                warnings.warn("Function return value was flattened.")

            if len(val) != len(np.unique(val)):
                if len(np.unique(val)) == self.shape:
                    return val
                else:
                    if self.target == "x":
                        raise ValueError(
                            f"Function returned duplicated values which is not allowed when"
                            " modifying the x axis. After filtering to unique values "
                            f"a {len(np.unique(val))}-length array was produced, "
                            f"but {self.shape} was expected."
                        )
                    return val

            if len(val) != self.shape:
                retval = self._broadcast(val)
                return retval
            return val
        else:
            raise TypeError("Function should return a number or Iterable type.")

    def _broadcast(self, val):
        if len(val) > self.shape:
            return val[:self.shape]
        elif len(val) < self.shape:
            if not self.shape % len(val) == 0:
                raise ValueError("Cannot broadcast safely to the desired shape.")
            else:
                return np.repeat(val, (self.shape % len(val)))
