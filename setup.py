import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Interferometry-PtrSkay",
    version="0.0.1",
    author="Péter Leéh",
    author_email="leeh123peter@gmail.com",
    description="UI for spectrally refined interferometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ptrskay3/Interferometry",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
    ],


)