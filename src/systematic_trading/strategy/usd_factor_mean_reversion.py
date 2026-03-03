import pandas as pd
import numpy as np
from data.minute_data_loader import get_minute_quotes
from sklearn.decomposition import PCA


def compute_mid_returns(minute_dict):

    mids = {pair: df["mid"] for pair, df in minute_dict.items()}
    mids = pd.DataFrame(mids).sort_index()

    logp = np.log(mids)
    rets = logp.diff()

    return rets.dropna()


def rolling_beta(y, x, window):
    # Fast rolling beta using covariance formula
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def compute_residuals(rets, usd_factor, window=120):
    residuals = pd.DataFrame(index=rets.index, columns=rets.columns)

    for pair in rets.columns:
        beta = rolling_beta(rets[pair], usd_factor, window)
        alpha = (
            rets[pair].rolling(window).mean() - beta * usd_factor.rolling(window).mean()
        )

        fitted = alpha + beta * usd_factor
        residuals[pair] = rets[pair] - fitted

    return residuals


class KalmanBeta:

    def __init__(self, delta=1e-4, R=1e-3):
        self.delta = delta
        self.R = R
        self.beta = 0.0
        self.P = 1.0

    def update(self, x, y):

        # Prediction
        beta_pred = self.beta
        P_pred = self.P + self.delta

        # Update
        K = P_pred * x / (x * P_pred * x + self.R)
        self.beta = beta_pred + K * (y - x * beta_pred)
        self.P = (1 - K * x) * P_pred

        return self.beta


def kalman_regression(pair_ret, usd_ret):

    kf = KalmanBeta()
    betas = []
    residuals = []

    for x, y in zip(usd_ret, pair_ret):

        beta = kf.update(x, y)
        betas.append(beta)

        residuals.append(y - beta * x)

    return (
        pd.Series(betas, index=pair_ret.index),
        pd.Series(residuals, index=pair_ret.index),
    )


def rolling_regression(pair_ret, usd_ret, window=500):
    """
    Compute rolling beta and residuals using
    rolling covariance / variance.
    """

    cov = pair_ret.rolling(window).cov(usd_ret)
    var = usd_ret.rolling(window).var()

    beta = cov / var
    alpha = pair_ret.rolling(window).mean() - beta * usd_ret.rolling(window).mean()
    fitted = alpha + beta * usd_ret

    residuals = pair_ret - fitted

    return beta, residuals


def generate_signals(residuals, lookback=500, zs_threshold=2, ret_threshold=0.005):

    cum_resid = residuals.cumsum()

    mean = cum_resid.rolling(lookback).mean()
    std = cum_resid.rolling(lookback).std()

    z = (cum_resid - mean) / std

    signal = pd.Series(0, index=z.index)

    signal[(z < -zs_threshold) & (cum_resid < -ret_threshold)] = 1
    signal[(z > zs_threshold) & (cum_resid > ret_threshold)] = -1

    return signal


def compute_executable_pnl(
    minute_dict, signals
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:

    pnl_dict = {}
    pnl_ex_cost_dict = {}
    trades_dict = {}

    for pair, df in minute_dict.items():

        bid = df["bid"]
        ask = df["ask"]
        mid = df["mid"]

        sig = signals[pair].reindex(df.index).fillna(0)

        position = sig.shift(1).fillna(0)

        ret = np.log(mid).diff()

        pnl = position * ret

        # Spread cost when position changes
        trade = position.diff().abs()

        spread = 0.5 * (ask - bid)
        cost = trade * (spread / mid)

        pnl_ex_cost_dict[pair] = pnl
        pnl_dict[pair] = pnl - cost
        trades_dict[pair] = trade

    pnl_df = pd.DataFrame(pnl_dict)
    pnl_ex_cost_df = pd.DataFrame(pnl_ex_cost_dict)
    portfolio_pnl = pnl_df.sum(axis=1)
    portfolio_pnl_ex_cost = pnl_ex_cost_df.sum(axis=1)
    trades_df = pd.DataFrame(trades_dict)

    return pnl_df, pnl_ex_cost_df, portfolio_pnl, portfolio_pnl_ex_cost, trades_df


def rolling_pca_usd_index(rets, window=2000):

    usd_series = []
    index = []

    prev_loading = None

    for i in range(window, len(rets)):

        window_data = rets.iloc[i-window:i]

        pca = PCA(n_components=1)
        pca.fit(window_data)

        loading = pca.components_[0]

        # Stabilize eigenvector sign
        if prev_loading is not None:
            if np.dot(loading, prev_loading) < 0:
                loading = -loading

        prev_loading = loading

        # Project current return (causal)
        current_ret = rets.iloc[i]
        usd_val = np.dot(current_ret, loading)

        usd_series.append(usd_val)
        index.append(rets.index[i])

    return pd.Series(usd_series, index=index, name="USD_index")


def usd_basket_index(rets):

    signed_rets = []

    for pair in rets.columns:

        if pair.startswith("USD"):
            signed_rets.append(rets[pair])
        else:
            signed_rets.append(-rets[pair])

    basket = pd.concat(signed_rets, axis=1).mean(axis=1)

    basket.name = "USD_index"

    return basket


def compute_usd_factor(rets, method="rolling_pca", window=2000):

    if method == "rolling_pca":
        return rolling_pca_usd_index(rets, window)

    elif method == "basket":
        return usd_basket_index(rets)

    else:
        raise ValueError("Unknown USD factor method")


def run_strategy(minute_dict: dict[str, pd.DataFrame], rolling_type: str):

    # Mid returns
    rets = compute_mid_returns(minute_dict)

    # USD factor
    usd_index, _ = compute_usd_factor(rets, method="rolling_pca")

    signals = {}

    for pair in rets.columns:

        if rolling_type == "kalman":
            beta, residuals = kalman_regression(rets[pair], usd_index)
        else:
            beta, residuals = rolling_regression(rets[pair], usd_index)

        sig = generate_signals(residuals)

        signals[pair] = sig

    signals = pd.DataFrame(signals)

    # Executable PnL
    pnl, pnl_ex_cost = compute_executable_pnl(minute_dict, signals)

    total_pnl = pnl.sum(axis=1)
    total_pnl_ex_cost = pnl_ex_cost.sum(axis=1)

    return total_pnl, total_pnl_ex_cost
