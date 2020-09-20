import types
from weakref import WeakSet

from pysprint.utils.decorators import pprint_disp

__all__ = ["_DatasetBase", "C_LIGHT"]

C_LIGHT = 299.792458


class _DatasetBase(type):
    """
    Base metaclass that defines the behaviour of any interferogram
    class, and also this sets up the registry.

    Set autoprinting if subclass implements calculate-like method.
    """
    def __init__(cls, name, bases, attrs):
        super(_DatasetBase, cls).__init__(name, bases, attrs)
        cls._instances = WeakSet()

    def __call__(cls, *args, **kwargs):
        inst = super(_DatasetBase, cls).__call__(*args, **kwargs)
        cls._instances.add(inst)
        return inst

    def _get_instances(cls, recursive=False):
        instances = list(cls._instances)
        if recursive:
            for child in cls.__subclasses__():
                instances += child._get_instances(recursive=recursive)
        return list(set(instances))

    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):
                if attr_name == "calculate" or attr_name.startswith("calculate"):
                    attrs[attr_name] = pprint_disp(attr_value)
            else:
                if attr_name == "calculate" or attr_name.startswith("calculate"):
                    if isinstance(attr_value, staticmethod):
                        attrs[attr_name] = staticmethod(pprint_disp(attr_value.__func__))

        return super(_DatasetBase, cls).__new__(cls, name, bases, attrs)
