"""
This code is mostly adapted from pandas/pandas/util/_print_versions.py.
pandas is licensed under the BSD 3-Clause "New" or "Revised" License.
See at: pandas.pydata.org
"""
import os
import sys
import platform
import struct
import importlib

__all__ = ["print_info"]

SUPPORTED_VERSIONS = {
    "matplotlib": "2.2.2",
    "scipy": "1.2.0",
    "numpy": "1.16.0",
    "pandas": "0.24.0",
    "lmfit": "1.0.1",
    "numba": "0.46.0",
    "IPython": "7.12.0",
    "pytest": "5.0.1",
}


def get_system_info():
    uname_result = platform.uname()
    return {
        "python": ".".join(str(j) for j in sys.version_info),
        "python-bits": struct.calcsize("P") * 8,
        "OS": uname_result.system,
        "OS-release": uname_result.release,
        "Version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
    }


def find_version(module):
    version = getattr(module, "__version__", None)
    if version is None:
        raise ImportError(f"Version for {module.__name__} can't be found.")
    return version


def import_optional(name):
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return module


def get_dep_info():
    deps = [
        "pysprint",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pytest",
        "lmfit",
        "numba",
        "IPython",
    ]
    deps.extend(list(SUPPORTED_VERSIONS))
    result = {}
    for modname in deps:
        mod = import_optional(modname)
        result[modname] = find_version(mod) if mod else None
    return result


def print_info():
    sysinfo = get_system_info()
    deps = get_dep_info()
    n = max(len(x) for x in deps)
    print("\nPYSPRINT ANALYSIS TOOL")
    print("\n        SYSTEM")
    print("----------------------")
    for k, v in sysinfo.items():
        print(f"{k:<{n + 1}}: {v}")
    print("\n      DEPENDENCY")
    print("----------------------")
    for k, v in deps.items():
        print(f"{k:<{n + 1}}: {v}")
    print("\n      ADDITIONAL")
    print("----------------------")
    try:
        ip = __IPYTHON__
    except NameError:
        ip = None
    nm = "IPython"
    is_spyder = any("SPYDER" in name for name in os.environ)
    _is_spyder = "Spyder"
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta", "history"))
    _is_conda = "Conda-env"
    print(f"{_is_conda:<{n + 1}}: {is_conda}")
    print(f"{nm:<{n + 1}}: {ip}")
    print(f"{_is_spyder:<{n + 1}}: {is_spyder}")
