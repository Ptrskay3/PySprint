from pysprint import config as c


def is_nonnegative_int(value):
    if isinstance(value, int) and value >= 0:
        return True
    raise ValueError


_precision_doc = """
: int
    The precision of floating point numbers in the output.
"""

c.register_config_value("precision", 5, doc=_precision_doc, validator=is_nonnegative_int)
