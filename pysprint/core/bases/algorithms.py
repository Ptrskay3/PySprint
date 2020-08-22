import numpy as np


def longest_common_subsequence(x1, y1, x2, y2, tol=None):
    """
    Given two datasets with x-y values, find the longest common
    subsequence of them including a small threshold which might be
    present due to numerical errors.
    This function is mainly used when two datasets's y values need
    to be multiplied together, but their domains are slightly off.

    Parameters
    ----------
    x1 : np.ndarray-like
        The x values for the original array
    y1 : np.ndarray-like
        The y values for the original array
    x2 : np.ndarray-like
        The x values for the second array
    y2 : np.ndarray-like
        The y values for the second array
    tol : float, optional
        The tolerance which determines how big difference is allowed
        _between x values to interpret them as the same datapoint.

    Returns
    -------
    longest_x1: np.ndarray
        The x values of longest common subsequence in x1.
    longest_y1: np.ndarray
        The y values of longest common subsequence in y1.
    longest_x2: np.ndarray
        The x values of longest common subsequence in x2.
    longest_y2: np.ndarray
        The y values of longest common subsequence in y2.
    """
    # sort both datasets first to maintain order
    idx1 = np.argsort(x1)
    x1, y1 = x1[idx1], y1[idx1]
    idx2 = np.argsort(x2)
    x2, y2 = x2[idx2], y2[idx2]

    if tol is None:
        tol = np.max(np.diff(x2))

    dist = np.abs(x1 - x2[:, None])
    i1 = np.argmin(dist, axis=1)
    i2 = np.flatnonzero(dist[np.arange(x2.size), i1] < tol)
    i1 = i1[i2]

    mask = (np.diff(i1) == 1) & (np.diff(i2) == 1)

    # smear the mask to include both endpoints
    mask = np.r_[False, mask] | np.r_[mask, False]

    # pad the mask to ensure proper indexing and find the changeover points
    locs = np.diff(np.r_[False, mask, False])
    inds = np.flatnonzero(locs)
    lengths = inds[1::2] - inds[::2]

    k = np.argmax(lengths)
    start = inds[2 * k]
    stop = inds[2 * k + 1]

    longest_x1 = x1[i1[start:stop]]
    longest_y1 = y1[i1[start:stop]]
    longest_x2 = x2[i2[start:stop]]
    longest_y2 = y2[i2[start:stop]]

    return longest_x1, longest_y1, longest_x2, longest_y2
