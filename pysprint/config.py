"""
The implementation is mostly taken from the pandas library with some simplifications
(e.g. deprecations are left out).
# Copyright (c) Pandas Development Team.
# Distributed under the terms of the BSD 3-Clause "New" or "Revised" License.
"""

import re
import keyword
import tokenize
from collections import namedtuple
from contextlib import ContextDecorator


class ConfigError(ValueError):
    pass


RegisteredConfigValue = namedtuple("RegisteredConfigValue", "key default doc validator")

_registered_config = {}

_global_config = {}


def _select_option(pattern):
    if pattern in _registered_config:
        return [pattern]

    keys = sorted(_registered_config.keys())

    return [k for k in keys if re.search(pattern, k, re.I)]


def _get_single_key(pattern):
    keys = _select_option(pattern)
    if len(keys) == 0:
        raise ConfigError(f"No such key(s): {repr(pattern)}")
    if len(keys) > 1:
        raise ConfigError("Pattern matched multiple keys")
    key = keys[0]
    return key


def _get_config_value(pattern):
    key = _get_single_key(pattern)
    return _global_config[key]


def _set_config_value(*args, **kwargs):
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    if kwargs:
        kwarg = list(kwargs.keys())[0]
        raise TypeError(f'_set_config_value() got an unexpected keyword argument "{kwarg}"')

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k)

        opt = _registered_config.get(key)
        if opt and opt.validator:
            opt.validator(v)

        _global_config[key] = v


def register_config_value(key, default, doc, validator=None):
    """
    Top level function which sets up the config.
    """
    key = key.lower()

    if key in _registered_config:
        raise ConfigError(f"Option '{key}' has already been registered.")

    if validator:
        validator(default)

    path = key.split(".")

    for k in path:
        if not re.match("^" + tokenize.Name + "$", k):
            raise ValueError(f"{k} is not a valid identifier.")
        if keyword.iskeyword(k):
            raise ValueError(f"{k} is a python keyword.")

    loc = _global_config
    msg = "'{option}' is already an option"

    for i, p in enumerate(path[:-1]):
        if not isinstance(loc, dict):
            raise ConfigError(msg.format(option=".".join(path[:i])))
        if p not in loc:
            loc[p] = {}
        loc = loc[p]

    if not isinstance(loc, dict):
        raise ConfigError(msg.format(option=".".join(path[:-1])))

    # initialize the default config
    loc[path[-1]] = default

    # write metadata
    _registered_config[key] = RegisteredConfigValue(
        key=key, default=default, doc=doc, validator=validator
    )


def _describe_config_value(pattern):

    keys = _select_option(pattern)
    if len(keys) == 0:
        raise ConfigError("No such key(s).")

    out = "\n".join([_build_config_value_description(k) for k in keys])

    return out


def _build_config_value_description(key):
    option = _registered_config.get(key)

    out = f"{key} "

    if option.doc:
        out += "\n".join(option.doc.strip().split("\n"))
    else:
        out += "No description."

    if option:
        out += f"\n value: {_get_config_value(key)} (defaults to {option.default})"
    return out


class setting(ContextDecorator):
    def __init__(self, *args):
        if not (len(args) % 2 == 0 and len(args) >= 2):
            raise ConfigError

        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self):
        self.restore = [(pattern, _get_config_value(pattern)) for pattern, val in self.ops]

        for pattern, val in self.ops:
            _set_config_value(pattern, val)

    def __exit__(self, *args):
        if self.restore:
            for pattern, val in self.restore:
                _set_config_value(pattern, val)
