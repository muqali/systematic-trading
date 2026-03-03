from strategy.strategy import Strategy
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class FactorMeanReversionStrategy(Strategy):
    def __init__(self, close_price_dict: dict[str, pd.DataFrame]):
        self.close_price_dict = close_price_dict

    def compute_mid_returns(self) -> pd.DataFrame:

        mids = {pair: df["mid"] for pair, df in self.close_price_dict.items()}
        mids = pd.DataFrame(mids).sort_index()

        logp = np.log(mids)
        rets = logp.diff()

        return rets.dropna()

    def rolling_regression(self, pair_ret, usd_ret, window=500):
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

    def rolling_pca(self, rets, window=2000) -> pd.Series:

        usd_series = []
        index = []

        prev_loading = None

        for i in range(window, len(rets)):

            window_data = rets.iloc[i - window : i]

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

    def usd_basket_index(self, rets) -> pd.Series:

        signed_rets = []

        for pair in rets.columns:

            if pair.startswith("USD"):
                signed_rets.append(rets[pair])
            else:
                signed_rets.append(-rets[pair])

        basket = pd.concat(signed_rets, axis=1).mean(axis=1)

        basket.name = "USD_index"

        return basket

    def compute_principal_factor(self, rets, method="rolling_pca", window=2000):

        if method == "rolling_pca":
            return self.rolling_pca(rets, window)

        elif method == "basket":
            return self.usd_basket_index(rets)

        else:
            raise ValueError("Unknown USD factor method")

    def generate_signals(self) -> dict[str, pd.Series]:

        # Mid returns
        rets = self.compute_mid_returns()

        # USD factor
        factor = self.compute_principal_factor(rets, method="rolling_pca")

        signals = {}

        lookback = 500
        zs_entry_threshold = 2.0
        zs_exit_threshold = 0.5
        ret_entry_threshold = 0.005
        ret_exit_threshold = 0.001

        for pair in rets.columns:

            beta, residuals = self.rolling_regression(rets[pair], factor)
            # beta, residuals = kalman_regression(rets[pair], usd_index)

            cum_resid = residuals.cumsum()

            mean = cum_resid.rolling(lookback).mean()
            std = cum_resid.rolling(lookback).std()

            z = (cum_resid - mean) / std

            signal = pd.Series(0, index=z.index, dtype=int)
            position = 0

            for ts in z.index:
                z_val = z.loc[ts]
                resid_val = cum_resid.loc[ts]

                if pd.isna(z_val) or pd.isna(resid_val):
                    signal.loc[ts] = 0
                    continue

                if position == 0:
                    # Entry conditions
                    if (z_val < -zs_entry_threshold) and (resid_val < -ret_entry_threshold):
                        position = 1
                    elif (z_val > zs_entry_threshold) and (resid_val > ret_entry_threshold):
                        position = -1
                elif position == 1:
                    # Exit long when residual has sufficiently mean-reverted
                    if (z_val > -zs_exit_threshold) or (resid_val > -ret_exit_threshold):
                        position = 0
                elif position == -1:
                    # Exit short when residual has sufficiently mean-reverted
                    if (z_val < zs_exit_threshold) or (resid_val < ret_exit_threshold):
                        position = 0

                signal.loc[ts] = position

            signals[pair] = signal

        return signals
