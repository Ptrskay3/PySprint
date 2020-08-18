import numpy as np
import pytest

from pysprint.core.bases.algorithms import longest_common_subsequence


def test_lcs_bases():
    x = np.arange(20)
    a, b, c, d = longest_common_subsequence(x, x, x, x)
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, x)
    np.testing.assert_array_equal(c, x)
    np.testing.assert_array_equal(d, x)


def test_lcs_nonbase_fail():
    x = np.arange(20)
    y = x + 0.1
    with pytest.raises(ValueError):
        longest_common_subsequence(x, x, y, y, tol=0.001)


def test_lcs_nonbase():
    x = np.arange(20)
    y = x + 0.1
    a, b, c, d = longest_common_subsequence(x, x, y, y)
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, x)
    np.testing.assert_array_equal(c, y)
    np.testing.assert_array_equal(d, y)
