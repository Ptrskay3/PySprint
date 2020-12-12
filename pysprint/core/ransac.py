"""
This is a utility module for RANSAC regression filtering. We use scikit-learn's RANSACRegressor class
with a custom polynomial model.

References:
https://en.wikipedia.org/wiki/Random_sample_consensus
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor

warnings.simplefilter('ignore', np.RankWarning)

def make_regression_class(degree):
    class PolynomialModel:
        def __init__(self, degree=degree, coeffs=None):
            self.degree = degree
            self.coeffs = coeffs

        def fit(self, X, y):
            self.coeffs = np.polyfit(X.ravel(), y, self.degree)

        def get_params(self, deep=False):
            return {'coeffs': self.coeffs}

        def set_params(self, coeffs=None, random_state=None):
            self.coeffs = coeffs

        def predict(self, X):
            poly_eqn = np.poly1d(self.coeffs)
            y_hat = poly_eqn(X.ravel())
            return y_hat

        def score(self, X, y):
            return mean_squared_error(y, self.predict(X))
    return PolynomialModel


def run_regressor(phase, degree, plot=True, **kwds):
    # this is the most important argument for us, so we isolate it
    residual_threshold = kwds.pop("residual_threshold", 0.5) * np.std(phase.y)

    # we must create the class on the fly, because sklearn copies the class,
    # and it uses the default degree param.
    model = make_regression_class(degree=degree)()
    ransac = RANSACRegressor(
        model,
        residual_threshold=residual_threshold,
        **kwds
    )
    ransac.fit(phase.x[:, np.newaxis], phase.y)
    inlier_mask = ransac.inlier_mask_

    y_hat = ransac.predict(phase.x[:, np.newaxis])

    if plot:
        _, ax = plt.subplots()
        ax.clear()
        phase.plot(ax=ax)
        ax.plot(phase.x[inlier_mask], phase.y[inlier_mask], "k+", label="inliers")
        ax.plot(phase.x, y_hat, 'g--', label='RANSAC estimated curve')
    return phase.x[inlier_mask], phase.y[inlier_mask]
