import numpy as np
from pysprint import Generator, WFTMethod


def test_basic():
    g = Generator(
        1,
        3,
        2,
        3000,
        GDD=400,
        TOD=4000,
        FOD=4000,
        QOD=50000,
        pulse_width=5,
        resolution=0.01
    )

    g.generate()

    f = WFTMethod(*g.data)
    f.add_window_linspace(1.25, 2.75, 350, fwhm=0.017)

    d, _, _ = f.calculate(reference_point=2, order=5)

    np.testing.assert_array_almost_equal(d, [3000.19708, 399.94187, 3998.03310, 3991.45904, 49894.99719], decimal=4)
