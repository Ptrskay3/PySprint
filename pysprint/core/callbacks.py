from pysprint.utils import run_from_ipython

__all__ = ["defaultcallback", "eager_executor"]


def defaultcallback(broadcaster, listener=None):
    """
    The default recorder for SPP data.
    """
    if listener is not None:
        listener._container[broadcaster] = broadcaster.emit()


def eager_executor(reference_point=None, order=None, cleanup=True, **kwds):
    """
    This function can be used as a `callback` function. It will
    constantly monitor for SPP data changes and will re-calculate
    the dispersion on any modification.

    Parameters
    ----------
    reference_point : float
        The reference point to use.
    order : int
        The dispersion order to look for.
    cleanup : bool, optional
        Whether to flush the output and override previous results.
    """
    if not reference_point or not order:
        raise RuntimeError(
            "Reference point and order is required."
        )

    def inner(broadcaster, listener=None):
        defaultcallback(broadcaster, listener)
        if run_from_ipython():
            from IPython.display import clear_output  # noqa
        else:
            import sys
            clear_output = sys.stdout.flush
        try:
            if cleanup:
                clear_output()
            listener.calculate(reference_point, order, **kwds)
        except (TypeError, ValueError):
            pass

    return inner
