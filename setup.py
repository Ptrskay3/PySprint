import sys
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

MAJOR = 0
MINOR = 0
MICRO = 29
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


setup(
    name="pysprint",
    version=VERSION,
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics"        
    ],
    install_requires=[
        'numpy',
        'scipy', 
        'matplotlib',
        'pandas', 
        'lmfit'
      ],
)
