import numpy as np

__all__ = ["correct_sign"]


def correct_sign(x, flip_increasing=True):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    increasing_idx = np.where(np.diff(x) > 0)
    # inc_len = len(increasing_idx[0])
    decreasing_idx = np.where(np.diff(x) < 0)
    # dec_len = len(decreasing_idx[0])
    # stationary_idx = np.where(np.diff(x) == 0)
    increasing = group_consecutives(increasing_idx)
    decreasing = group_consecutives(decreasing_idx)
    if not (len(increasing) == 1 and len(decreasing) == 1):
        raise ValueError(
            "Values could not be split into two " "strictly monotonic parts safely."
        )
    if flip_increasing:
        x[increasing_idx[0][0]] *= -1
        x[increasing_idx[0] + 1] *= -1
        return x
    x[decreasing_idx[0]] *= -1
    return x


def group_consecutives(arr, step=1):
    return np.split(arr, np.where(np.diff(arr) != step)[0] + 1)
