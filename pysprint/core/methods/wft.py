from pysprint.core.bases.dataset import Dataset


class WFTMethod(Dataset):
    """Basic interface for Windowed Fourier Transfrom Method."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def add_window_sequence(self, start, stop, step=1, scale='linear'):
        """
        Build a window sequence of given parameters to apply on ifg.
        """
        if scale not in ('linear', 'geometric'):
            raise ValueError

