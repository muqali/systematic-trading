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
        instruments: tuple[str],
        close_price_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict[str, float] = {
            "zs_entry_threshold": 2.0,
            "zs_exit_threshold": 0.0,
            "spread_entry_multiplier": 10.0,
            "residual_rolling_lookback": pd.Timedelta(hours=5),
        },
    ):
        self.instruments = instruments
        self.close_price_dict = close_price_dict
        self.zs_entry_threshold = hyper_param_dict.get("zs_entry_threshold")
        self.zs_exit_threshold = hyper_param_dict.get("zs_exit_threshold")
        self.spread_entry_multiplier = hyper_param_dict.get("spread_entry_multiplier")
        self.residual_rolling_lookback = hyper_param_dict.get(
            "residual_rolling_lookback"
        )

    def compute_mid_returns(self) -> pd.DataFrame:

        mids = {pair: df["mid"] for pair, df in self.close_price_dict.items()}
        mids = pd.DataFrame(mids).sort_index()

        logp = np.log(mids)
        rets = logp.diff()

        return rets.dropna()

    def rolling_regression(self, pair_ret, factor_ret, window=pd.Timedelta(14)):
        """
        Compute rolling beta and residuals using
        rolling covariance / variance.
        """

        df = pd.concat({"pair": pair_ret, "factor": factor_ret}, axis=1)
        df["factor"] = df["factor"].fillna(0.0)
        pair_ret = df["pair"]
        factor_ret = df["factor"]

        cov = pair_ret.rolling(window=window).cov(factor_ret)
        var = factor_ret.rolling(window=window).var()

        beta = cov / var
        alpha = (
            pair_ret.rolling(window=window).mean()
            - beta * factor_ret.rolling(window).mean()
        )
        fitted = alpha + beta * factor_ret

        residuals = pair_ret - fitted

        return beta, residuals

    def calculate_rolling_pca_loadings(
        self,
        usd_rets,
        window=pd.Timedelta(days=60),
        calc_freq=pd.Timedelta(hours=6),
        n_components=1,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

        if isinstance(calc_freq, float):
            calc_freq = window * calc_freq

        agg_rets = (
            usd_rets.resample(calc_freq, label="right", closed="right")
            .sum()
            .dropna(how="all")
        )

        loadings_dict = {f"PC{i+1}": [] for i in range(n_components)}
        explained_var_dict = {f"PC{i+1}": [] for i in range(n_components)}
        index = []

        idx = agg_rets.index
        names = list(agg_rets.keys())

        prev_loadings = None
        last_calc_time = None

        for i in range(1, len(agg_rets)):
            t = idx[i]
            start_t = t - window
            left = idx.searchsorted(start_t, side="left")

            window_data = agg_rets.iloc[left:i]

            # Only run PCA when full window is available
            if window_data.empty or (t - window_data.index[0] < window):
                continue

            # Check if we should calculate PCA at this timestamp
            should_calculate = (
                last_calc_time is None or (t - last_calc_time) >= calc_freq
            )

            if should_calculate:
                X = mad_clip(window_data, z=5.0)

                pca = PCA(n_components=n_components)
                pca.fit(X)

                current_loadings = pca.components_  # Shape: (n_components, n_features)
                explained_variance_ratio = (
                    pca.explained_variance_ratio_
                )  # Shape: (n_components,)

                # Stabilize eigenvector sign
                if prev_loadings is not None:
                    for comp_idx in range(n_components):
                        if (
                            np.dot(current_loadings[comp_idx], prev_loadings[comp_idx])
                            < 0
                        ):
                            current_loadings[comp_idx] = -current_loadings[comp_idx]

                prev_loadings = current_loadings.copy()
                last_calc_time = t

                # Store loadings for each component
                for comp_idx in range(n_components):
                    pc_name = f"PC{comp_idx+1}"
                    loadings_dict[pc_name].append(current_loadings[comp_idx])
                    explained_var_dict[pc_name].append(
                        explained_variance_ratio[comp_idx]
                    )

                index.append(t)

        # Create loadings DataFrames (one per component)
        loadings_df = {
            pc_name: pd.DataFrame(loadings_dict[pc_name], index=index, columns=names)
            for pc_name in loadings_dict.keys()
        }

        # Create explained variance DataFrame
        explained_var_df = pd.DataFrame(explained_var_dict, index=index)

        return loadings_df, explained_var_df

    def rolling_pca(
        self,
        rets,
        window=pd.Timedelta(days=60),
        calc_freq=pd.Timedelta(hours=6),
        n_components=1,
    ) -> pd.DataFrame:

        usd_rets = rets.filter(regex="^(?!.*UDX).*USD")

        loadings_by_pc, _ = self.calculate_rolling_pca_loadings(
            usd_rets, window=window, calc_freq=calc_freq, n_components=n_components
        )

        usd_series_by_pc: dict[str, pd.Series] = {}
        for pc_name, pc_loadings in loadings_by_pc.items():
            if pc_loadings.empty:
                continue
            aligned_loadings = pc_loadings.reindex(usd_rets.index, method="ffill")
            usd_series_by_pc[pc_name] = (usd_rets * aligned_loadings).sum(axis=1)

        usd_series_df = pd.DataFrame(usd_series_by_pc, index=usd_rets.index)
        usd_series_df = usd_series_df.dropna(how="all")

        return usd_series_df

    def customised_usd_basket(self, rets) -> pd.Series:

        signed_rets = []

        for pair in rets.columns:

            if pair.startswith("USD"):
                signed_rets.append(rets[pair])
            else:
                signed_rets.append(-rets[pair])

        basket = pd.concat(signed_rets, axis=1).mean(axis=1)

        basket.name = "USD_Basket"

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
        self, rets, method="rolling_pca", window=pd.Timedelta(days=60)
    ):
        cache_key = self._principal_factor_cache_key(rets, method, window)
        cached = self.__class__._principal_factor_cache.get(cache_key)
        if cached is not None:
            return cached

        if method == "rolling_pca":
            factor_df = self.rolling_pca(rets, window)
            factor = (
                factor_df["PC1"] if "PC1" in factor_df.columns else factor_df.iloc[:, 0]
            )

        elif method == "basket":
            factor = self.customised_usd_basket(rets)
        elif method == "dxy":
            factor = rets["UDXUSD"]

        else:
            raise ValueError("Unknown USD factor method")

        self.__class__._principal_factor_cache[cache_key] = factor
        return factor

    def generate_signals(self) -> dict[str, pd.Series]:

        factor_pca_rolling_lookback = pd.Timedelta(days=60)
        spread_rolling_lookback = pd.Timedelta(minutes=30)

        # Mid returns
        rets = self.compute_mid_returns()

        # USD factor
        factor = self.compute_principal_factor(
            rets, method="rolling_pca", window=factor_pca_rolling_lookback
        )

        signals = {}

        for pair in self.instruments:

            beta, residuals = self.rolling_regression(
                rets[pair], factor, window=factor_pca_rolling_lookback
            )
            # beta, residuals = kalman_regression(rets[pair], usd_index)

            # cum_resid = residuals.cumsum()
            cum_resid = residuals.rolling(window=self.residual_rolling_lookback).sum()

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

            signal = pd.Series(0, index=z.index, dtype=int)
            position = 0

            for ts in z.index:
                z_val = z.loc[ts]
                resid_val = cum_resid.loc[ts]
                entry_th = ret_entry_threshold.loc[ts]

                if pd.isna(z_val) or pd.isna(resid_val) or pd.isna(entry_th):
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
                    if z_val > -self.zs_exit_threshold:
                        position = 0
                elif position == -1:
                    # Exit short when residual has sufficiently mean-reverted
                    if z_val < self.zs_exit_threshold:
                        position = 0

                signal.loc[ts] = position

            signals[pair] = signal

        return signals

    def generate_signals2(self) -> dict[str, pd.Series]:

        factor_pca_rolling_lookback = pd.Timedelta(days=60)
        spread_rolling_lookback = pd.Timedelta(minutes=30)

        # Mid returns
        rets = self.compute_mid_returns()

        # USD factor and loadings
        factor = self.compute_principal_factor(
            rets, method="rolling_pca", window=factor_pca_rolling_lookback
        )
        loadings_by_pc, _ = self.calculate_rolling_pca_loadings(
            rets.filter(regex="^(?!.*UDX).*USD"),
            window=factor_pca_rolling_lookback,
            calc_freq=pd.Timedelta(hours=12),
            n_components=2,
        )

        signals: dict[str, pd.Series] = {}
        hedge_positions: dict[str, pd.Series] = {
            pair: pd.Series(0.0, index=rets.index, dtype=float) for pair in rets.columns
        }

        for pair in self.instruments:

            beta, residuals = self.rolling_regression(
                rets[pair], factor, window=factor_pca_rolling_lookback
            )

            cum_resid = residuals.rolling(window=self.residual_rolling_lookback).sum()

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

            signal = pd.Series(0, index=z.index, dtype=int)
            position = 0

            for ts in z.index:
                z_val = z.loc[ts]
                resid_val = cum_resid.loc[ts]
                entry_th = ret_entry_threshold.loc[ts]

                if pd.isna(z_val) or pd.isna(resid_val) or pd.isna(entry_th):
                    signal.loc[ts] = 0
                    continue

                if position == 0:
                    if (z_val < -self.zs_entry_threshold) and (resid_val < -entry_th):
                        position = 1
                    elif (z_val > self.zs_entry_threshold) and (resid_val > entry_th):
                        position = -1
                elif position == 1:
                    if z_val > -self.zs_exit_threshold:
                        position = 0
                elif position == -1:
                    if z_val < self.zs_exit_threshold:
                        position = 0

                signal.loc[ts] = position

            signals[pair] = signal

            beta_aligned = beta.reindex(signal.index)
            scale = (
                (-signal * beta_aligned).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )

            # Apply hedge in each factor component using its loadings
            for pc_name, pc_loadings in loadings_by_pc.items():
                if pc_loadings.empty:
                    continue
                aligned_loadings = pc_loadings.reindex(rets.index, method="ffill")
                for hedge_pair in aligned_loadings.columns:
                    hedge_positions[hedge_pair] = hedge_positions[hedge_pair].add(
                        scale * aligned_loadings[hedge_pair], fill_value=0.0
                    )

        for pair, hedge_series in hedge_positions.items():
            signals[pair] = signals.get(pair, pd.Series(0, index=rets.index)).add(
                hedge_series, fill_value=0.0
            )
        return signals
