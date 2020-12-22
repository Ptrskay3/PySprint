from glob import glob

import numpy as np

__all__ = ["correct_sign", "from_pat"]


# TODO: rewrite or deprecate this.
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
            "Values could not be split into two strictly monotonic parts safely."
        )
    if flip_increasing:
        x[increasing_idx[0][0]] *= -1
        x[increasing_idx[0] + 1] *= -1
        return x
    x[decreasing_idx[0]] *= -1
    return x


def group_consecutives(arr, step=1):
    return np.split(arr, np.where(np.diff(arr) != step)[0] + 1)


def from_pat(pat, mod=None):
    if mod is None:
        return {'ifg_names': glob(pat)}
    elif mod == 3:
        files = sorted(glob(pat))
        return {
            'ifg_names': files[::mod],
            'sam_names': files[1::mod],
            'ref_names': files[2::mod]
        }
    elif mod == -1:
        files = sorted(glob(pat))
        return {
            'ifg_names': files[::mod]
        }
    else:
        raise ValueError(
            f"mod = {mod} is ambigous. Consider using your own logic for selecting files."
        )
    