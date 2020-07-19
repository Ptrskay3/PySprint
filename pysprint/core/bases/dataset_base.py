import types

from pysprint.utils import print_disp

__all__ = ["DatasetBase", "C_LIGHT"]

C_LIGHT = 299.792458


class DatasetBase(type):
    """Base metaclass that defines the behaviour
    of any interferogram class.

    Set autoprinting if subclass implements calculate-like method.
    """

    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):
                if attr_name == "calculate" or attr_name.startswith("calculate"):
                    attrs[attr_name] = print_disp(attr_value)
            else:
                if attr_name == "calculate" or attr_name.startswith("calculate"):
                    if isinstance(attr_value, staticmethod):
                        attrs[attr_name] = staticmethod(print_disp(attr_value.__func__))
        return super(DatasetBase, cls).__new__(cls, name, bases, attrs)
