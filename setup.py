import os
import sys


if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as SdistCommand


with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    from setuptools_rust import RustExtension, Binding
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, "-m", "pip", "install", "setuptools-rust"])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension, Binding

from versioneer import get_cmdclass, get_version

cmdclass = get_cmdclass()
_VersioneerSdist = cmdclass['sdist']

class CargoModifiedSdist(_VersioneerSdist):
    """Modifies Cargo.toml to use an absolute rather than a relative path

    The current implementation of PEP 517 in pip always does builds in an
    isolated temporary directory. This causes problems with the build, because
    Cargo.toml necessarily refers to the current version of pyo3 by a relative
    path.

    Since these sdists are never meant to be used for anything other than
    tox / pip installs, at sdist build time, we will modify the Cargo.toml
    in the sdist archive to include an *absolute* path to pyo3.
    """

    def make_release_tree(self, base_dir, files):
        """Stages the files to be included in archives"""
        super().make_release_tree(base_dir, files)
        import toml

        # Cargo.toml is now staged and ready to be modified
        cargo_loc = os.path.join(base_dir, "Cargo.toml")
        assert os.path.exists(cargo_loc)

        with open(cargo_loc, "r") as f:
            cargo_toml = toml.load(f)

        rel_pyo3_path = cargo_toml["dependencies"]["pyo3"]["path"]
        base_path = os.path.dirname(__file__)
        abs_pyo3_path = os.path.abspath(os.path.join(base_path, rel_pyo3_path))
        cargo_toml["dependencies"]["pyo3"]["path"] = abs_pyo3_path

        with open(cargo_loc, "w") as f:
            toml.dump(cargo_toml, f)

cmdclass["sdist"] = CargoModifiedSdist

setup(
    name="pysprint",
    version=get_version(),
    cmdclass=cmdclass,
    author="Péter Leéh",
    author_email="leeh123peter@gmail.com",
    description="Spectrally refined interferometry for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ptrskay3/PySprint",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["numpy>=1.16.6", "scipy", "matplotlib", "pandas", "Jinja2", "scikit-learn"],
    setup_requires=["setuptools-rust>=0.10.1", "wheel"],
    extras_require={"optional": ["numba", "lmfit", "pytest", "dask"]},
    rust_extensions=[
        RustExtension("pysprint.numerics", "Cargo.toml", debug=False, binding=Binding.PyO3),
        ],
    entry_points={
        'console_scripts': [
            'pysprint = pysprint.templates.build:main',
        ],
    },
    zip_safe=False,
)
