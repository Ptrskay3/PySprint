import numpy as np


__all__ = [
    "cos_fit1",
    "cos_fit2",
    "cos_fit3",
    "cos_fit4",
    "cos_fit5",
    "cos_fit6",
    "poly1",
    "poly2",
    "poly3",
    "poly4",
    "poly5",
    "poly6",
    "_fit_config",
    "_cosfit_config",
]


def poly6(x, b0, b1, b2, b3, b4, b5, b6):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6
    b4 = FOD / 24
    b5 = QOD / 120
    b6 = SOD / 720
    """
    return (
        b0
        + b1 * x
        + b2 * x ** 2
        + b3 * x ** 3
        + b4 * x ** 4
        + b5 * x ** 5
        + b6 * x ** 6
    )


def poly5(x, b0, b1, b2, b3, b4, b5):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6
    b4 = FOD / 24
    b5 = QOD / 120
    """
    return b0 + b1 * x + b2 * x ** 2 + b3 * x ** 3 + b4 * x ** 4 + b5 * x ** 5


def poly4(x, b0, b1, b2, b3, b4):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6
    b4 = FOD / 24
    """
    return b0 + b1 * x + b2 * x ** 2 + b3 * x ** 3 + b4 * x ** 4


def poly3(x, b0, b1, b2, b3):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    b3 = TOD / 6

    """
    return b0 + b1 * x + b2 * x ** 2 + b3 * x ** 3


def poly2(x, b0, b1, b2):
    """
    Taylor polynomial for fit
    b1 = GD
    b2 = GDD / 2
    """
    return b0 + b1 * x + b2 * x ** 2


def poly1(x, b0, b1):
    """
    Taylor polynomial for fit
    b1 = GD
    """
    return b0 + b1 * x


def cos_fit1(x, c0, c1, b0, b1):
    return c0 + c1 * np.cos(poly1(x, b0, b1))


def cos_fit2(x, c0, c1, b0, b1, b2):
    return c0 + c1 * np.cos(poly2(x, b0, b1, b2))


def cos_fit3(x, c0, c1, b0, b1, b2, b3):
    return c0 + c1 * np.cos(poly3(x, b0, b1, b2, b3))


def cos_fit4(x, c0, c1, b0, b1, b2, b3, b4):
    return c0 + c1 * np.cos(poly4(x, b0, b1, b2, b3, b4))


def cos_fit5(x, c0, c1, b0, b1, b2, b3, b4, b5):
    return c0 + c1 * np.cos(poly5(x, b0, b1, b2, b3, b4, b5))


def cos_fit6(x, c0, c1, b0, b1, b2, b3, b4, b5, b6):
    return c0 + c1 * np.cos(poly6(x, b0, b1, b2, b3, b4, b5, b6))


_fit_config = {1: poly1, 2: poly2, 3: poly3, 4: poly4, 5: poly5}


_cosfit_config = {
    1: cos_fit1,
    2: cos_fit2,
    3: cos_fit3,
    4: cos_fit4,
    5: cos_fit5,
}
