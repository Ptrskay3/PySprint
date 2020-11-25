from pysprint import config as cfg


def is_nonnegative_int(value):
    if isinstance(value, int) and value >= 0:
        return True
    raise ValueError


def is_normalized_float(value):
    if isinstance(value, (float, int)):
        if 0 <= value <= 1:
            return True
        raise ValueError
    raise ValueError


_precision_doc = """
: int
    The precision of floating point numbers in the output.
"""

cfg.register_config_value("precision", 5, doc=_precision_doc, validator=is_nonnegative_int)

_verbosity_doc = """
: int
    The verbosity level of logging outputs when using eager_executor.
"""

cfg.register_config_value("verbosity", 0, doc=_verbosity_doc, validator=is_nonnegative_int)

_scan_threshold_doc = """
: float
    Determines the maximum relative distance between any calculated dispersion coefficient.
    If the relative distance is bigger than this value, the two calculated values will be
    treated as distinct: the two sides won't be averaged out.
"""

cfg.register_config_value("scan_threshold", 0.5, doc=_scan_threshold_doc, validator=is_normalized_float)
