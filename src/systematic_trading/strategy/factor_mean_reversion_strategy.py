from strategy.strategy import Strategy
from sklearn.decomposition import PCA
from util.math_util import mad_clip
import pandas as pd
import numpy as np
from typing import Union


class FactorMeanReversionStrategy(Strategy):
    _principal_factor_cache: dict[tuple, pd.Series] = {}

    def __init__(
        self,
        close_price_dict: dict[str, pd.DataFrame],
        zs_entry_threshold: float = 2.0,
        zs_exit_threshold: float = 0.5,
        spread_entry_multiplier: float = 10.0,
        spread_exit_multiplier: float = 2.0,
    ):
        self.close_price_dict = close_price_dict
        self.zs_entry_threshold = zs_entry_threshold
        self.zs_exit_threshold = zs_exit_threshold
        self.spread_entry_multiplier = spread_entry_multiplier
        self.spread_exit_multiplier = spread_exit_multiplier

    def compute_mid_returns(self) -> pd.DataFrame:

        mids = {pair: df["mid"] for pair, df in self.close_price_dict.items()}
        mids = pd.DataFrame(mids).sort_index()

        logp = np.log(mids)
        rets = logp.diff()

        return rets.dropna()

    def rolling_regression(self, pair_ret, usd_ret, window=pd.Timedelta(14)):
        """
        Compute rolling beta and residuals using
        rolling covariance / variance.
        """

        df = pd.concat({"pair": pair_ret, "usd": usd_ret}, axis=1)
        df["usd"] = df["usd"].fillna(0.0)
        pair_ret = df["pair"]
        usd_ret = df["usd"]

        cov = pair_ret.rolling(window=window).cov(usd_ret)
        var = usd_ret.rolling(window=window).var()

        beta = cov / var
        alpha = (
            pair_ret.rolling(window=window).mean()
            - beta * usd_ret.rolling(window).mean()
        )
        fitted = alpha + beta * usd_ret

        residuals = pair_ret - fitted

        return beta, residuals

    def rolling_pca(
        self, rets, window=pd.Timedelta(days=14)
    ) -> tuple[pd.Series, pd.DataFrame]:

        usd_series, index, loadings = [], [], []
        idx = rets.index
        names = list(rets.keys())

        prev_loading = None

        for i in range(1, len(rets)):

            t = idx[i]
            start_t = t - window
            left = idx.searchsorted(start_t, side="left")

            window_data = rets.iloc[left:i]

            # Only run PCA when full window is available
            if window_data.empty or (t - window_data.index[0] < window):
                continue

            X = mad_clip(window_data, z=5.0)

            pca = PCA(n_components=1)
            pca.fit(X)

            loading = pca.components_[0]

            # Stabilize eigenvector sign
            if prev_loading is not None:
                if np.dot(loading, prev_loading) < 0:
                    loading = -loading

            prev_loading = loading

            # Project current return (causal)
            current_ret = rets.iloc[i]
            usd_val = np.dot(current_ret, loading)

            loadings.append(loading)
            usd_series.append(usd_val)
            index.append(rets.index[i])

        return pd.Series(usd_series, index=index, name="USD_index"), pd.DataFrame(
            loadings, index=index, columns=names
        )

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

    def _principal_factor_cache_key(
        self, rets: pd.DataFrame, method: str, window: Union[str, pd.Timedelta]
    ) -> tuple:
        if len(rets.index) == 0:
            first_ts = None
            last_ts = None
        else:
            first_ts = rets.index[0]
            last_ts = rets.index[-1]

        window_key = str(pd.Timedelta(window))

        return (
            id(self.close_price_dict),
            method,
            window_key,
            rets.shape,
            tuple(rets.columns),
            first_ts,
            last_ts,
        )

    def compute_principal_factor(
        self, rets, method="rolling_pca", window=pd.Timedelta(days=14)
    ):
        cache_key = self._principal_factor_cache_key(rets, method, window)
        cached = self.__class__._principal_factor_cache.get(cache_key)
        if cached is not None:
            return cached

        if method == "rolling_pca":
            factor = self.rolling_pca(rets, window)[0]

        elif method == "basket":
            factor = self.usd_basket_index(rets)

        else:
            raise ValueError("Unknown USD factor method")

        self.__class__._principal_factor_cache[cache_key] = factor
        return factor

    def generate_signals(self) -> dict[str, pd.Series]:

        factor_pca_rolling_lookback = pd.Timedelta(days=14)
        residual_rolling_lookback = pd.Timedelta(hours=5)
        spread_rolling_lookback = pd.Timedelta(minutes=120)

        # Mid returns
        rets = self.compute_mid_returns()

        # USD factor
        factor = self.compute_principal_factor(
            rets, method="rolling_pca", window=factor_pca_rolling_lookback
        )

        signals = {}

        for pair in rets.columns:

            beta, residuals = self.rolling_regression(
                rets[pair], factor, window=factor_pca_rolling_lookback
            )
            # beta, residuals = kalman_regression(rets[pair], usd_index)

            # cum_resid = residuals.cumsum()
            cum_resid = residuals.rolling(window=residual_rolling_lookback).sum()

            mean = cum_resid.rolling(window=factor_pca_rolling_lookback).mean()
            std = cum_resid.rolling(window=factor_pca_rolling_lookback).std()

            z = (cum_resid - mean) / std
            pair_df = self.close_price_dict[pair]
            spread_ret = ((pair_df["ask"] - pair_df["bid"]) / pair_df["mid"]).replace(
                [np.inf, -np.inf], np.nan
            )

            ret_entry_threshold = (
                spread_ret.rolling(window=spread_rolling_lookback).mean()
                * self.spread_entry_multiplier
            ).reindex(z.index)
            ret_exit_threshold = (
                spread_ret.rolling(window=spread_rolling_lookback).mean()
                * self.spread_exit_multiplier
            ).reindex(z.index)

            signal = pd.Series(0, index=z.index, dtype=int)
            position = 0

            for ts in z.index:
                z_val = z.loc[ts]
                resid_val = cum_resid.loc[ts]
                entry_th = ret_entry_threshold.loc[ts]
                exit_th = ret_exit_threshold.loc[ts]

                if (
                    pd.isna(z_val)
                    or pd.isna(resid_val)
                    or pd.isna(entry_th)
                    or pd.isna(exit_th)
                ):
                    signal.loc[ts] = 0
                    continue

                if position == 0:
                    # Entry conditions
                    if (z_val < -self.zs_entry_threshold) and (resid_val < -entry_th):
                        position = 1
                    elif (z_val > self.zs_entry_threshold) and (resid_val > entry_th):
                        position = -1
                elif position == 1:
                    # Exit long when residual has sufficiently mean-reverted
                    if (z_val > -self.zs_exit_threshold) or (resid_val > -exit_th):
                        position = 0
                elif position == -1:
                    # Exit short when residual has sufficiently mean-reverted
                    if (z_val < self.zs_exit_threshold) or (resid_val < exit_th):
                        position = 0

                signal.loc[ts] = position

            signals[pair] = signal

        return signals
