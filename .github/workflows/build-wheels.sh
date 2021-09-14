#!/bin/bash
set -ex

for PYBIN in /opt/python/cp3[6789]*/bin; do
    "${PYBIN}/pip" install maturin setuptools_rust twine
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release
done

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done