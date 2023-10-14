import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def sigmoid(xs, L, x0, k, b):
    y = L / (1 + np.exp(-k * (xs - x0))) + b
    return y


def fit_model(df, method):
    hours = df.columns.astype(np.float64)

    def fit_curve(ydata):
        ydata = np.array(ydata).astype(np.float64)

        # https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
        p0 = [max(ydata), np.median(hours), 1, min(ydata)]

        try:
            params, covariance = curve_fit(sigmoid, hours, ydata, p0, method=method)
        except RuntimeError:
            print('Runtime error during curve fitting. 0s will be used as fallback params')
            params = [0, 0, 0, 0]
        return pd.Series(np.round(params, 4), index=['L', 'x0', 'k', 'b'])

    return df.apply(fit_curve, axis=1)
