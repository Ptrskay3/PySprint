import os
import sys
import warnings
import platform
import struct
import importlib
import distutils

import pysprint

__all__ = ["print_info"]

SUPPORTED_VERSIONS = {
    "matplotlib": "2.2.2",
    "scipy": "1.2.0",
    "numpy": "1.16.0",
    "pandas": "0.24.0",
    "lmfit": "1.0.1",
    "numba": "0.46.0",
    "IPython": "7.12.0",
    "pytest": "5.0.1"
}


def get_system_info():
    uname_result = platform.uname()
    return {
        "python": ".".join(str(i) for i in sys.version_info),
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
    "IPython"
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
    print("\n  SYSTEM INFORMATION")
    print("----------------------")
    for k, v in sysinfo.items():
        print(f"{k:<{n + 1}}: {v}")
    print("\n    DEPENDENCY INFO")
    print("----------------------")
    for k, v in deps.items():
        print(f"{k:<{n + 1}}: {v}")
    print("\n      ADDITIONAL")
    print("----------------------")
    try:
        ip = __IPYTHON__
    except NameError:
        ip = None
    spy = any("SPYDER" in name for name in os.environ)
    nm = "IPython"
    is_spyder = "in Spyder"
    _is_conda = "Conda env"
    is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta', 'history'))
    print(f"{_is_conda:<{n + 1}}: {is_conda}")
    print(f"{nm:<{n + 1}}: {ip}")
    print(f"{is_spyder:<{n + 1}}: {spy}")