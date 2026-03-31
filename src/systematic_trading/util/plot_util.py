import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def plot_series_scatter(
    x: pd.Series, y: pd.Series, xlabel=None, ylabel=None, title="Scatter Plot"
):
    aligned = pd.concat([x.rename(xlabel), y.rename(ylabel)], axis=1).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(aligned[xlabel], aligned[ylabel], alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_ccf(x: pd.Series, y: pd.Series, nlags=40, alpha=0.05):

    fit = sm.graphics.tsa.plot_ccf(
        x, y, lags=np.arange(-nlags, nlags + 1), negative_lags=True, alpha=0.05
    )
    return fit
