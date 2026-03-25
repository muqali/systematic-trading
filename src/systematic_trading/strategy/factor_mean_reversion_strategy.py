from strategy.strategy import Strategy
from sklearn.decomposition import PCA
from util.math_util import mad_clip
import pandas as pd
import numpy as np
from typing import Union


class FactorMeanReversionStrategy(Strategy):
    _principal_factor_cache: dict[tuple, pd.Series] = {}

    @classmethod
    def _is_after_4pm_on_friday(cls, ts: pd.Timestamp) -> bool:
        return ts.dayofweek == 4 and ts.hour >= 16

    def __init__(
        self,
        instruments: tuple[str],
        close_price_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict[str, float] = {
            "zs_entry_threshold": 2.0,
            "zs_exit_threshold": 0.0,
            "spread_entry_multiplier": 10.0,
            "residual_rolling_lookback": pd.Timedelta(hours=5),
            "n_components": 2
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
        self.n_components = hyper_param_dict.get("n_components")

    def compute_mid_returns(self) -> pd.DataFrame:

        mids = {pair: df["mid"] for pair, df in self.close_price_dict.items()}
        mids = pd.DataFrame(mids).sort_index()

        logp = np.log(mids)
        rets = logp.diff()

        return rets.dropna()

    def rolling_regression(
        self,
        pair_ret,
        factors_ret,
        window=pd.Timedelta(days=60),
        return_bucket_length=pd.Timedelta(hours=6),
        calc_freq=pd.Timedelta(days=7),
    ):
        """
        Compute rolling multi-factor OLS exposures and residuals.
        """

        if isinstance(pair_ret, pd.DataFrame):
            if pair_ret.shape[1] != 1:
                raise ValueError("pair_ret must be a Series or single-column DataFrame")
            pair_ret = pair_ret.iloc[:, 0]
        if isinstance(factors_ret, pd.Series):
            factors_ret = factors_ret.to_frame()
        elif not isinstance(factors_ret, pd.DataFrame):
            raise ValueError("factors_ret must be a Series or DataFrame")

        if isinstance(return_bucket_length, float):
            return_bucket_length = window * return_bucket_length
        if isinstance(calc_freq, float):
            calc_freq = window * calc_freq

        factors_ret = factors_ret.copy()
        df = pd.concat([pair_ret.rename("pair"), factors_ret], axis=1).sort_index()
        df = df.dropna(subset=["pair"])

        pair_ret = df["pair"]
        factors_ret = df[factors_ret.columns]
        factor_cols = list(factors_ret.columns)

        beta_calc = pd.DataFrame([], columns=factor_cols, dtype=float)
        alpha_calc = pd.Series(dtype=float)
        idx = pair_ret.index
        last_checked_friday = None

        for i in range(len(idx)):
            t = idx[i]
            friday_key = t.normalize() if self._is_after_4pm_on_friday(t) else None
            if friday_key is None or friday_key == last_checked_friday:
                continue
            last_checked_friday = friday_key

            should_calculate = (
                beta_calc.empty
                or alpha_calc.empty
                or (t - alpha_calc.index[-1]) >= calc_freq
            )
            if not should_calculate:
                continue

            pair_until_t = pair_ret.loc[:t]
            factors_until_t = factors_ret.loc[:t]
            agg_pair = pair_until_t.resample(
                return_bucket_length, label="right", closed="right"
            ).sum(min_count=1)
            agg_factors = factors_until_t.resample(
                return_bucket_length, label="right", closed="right"
            ).sum(min_count=1)
            agg_df = pd.concat([agg_pair.rename("pair"), agg_factors], axis=1).dropna(
                subset=["pair"]
            )
            if agg_df.empty:
                continue
            agg_df[factor_cols] = agg_df[factor_cols].fillna(0.0)

            start_t = t - window - calc_freq
            left = agg_df.index.searchsorted(start_t, side="left")
            right = agg_df.index.searchsorted(t, side="right")
            window_data = agg_df.iloc[left:right]

            if window_data.empty or (t - window_data.index[0] < window):
                continue

            y = window_data["pair"]
            X = window_data[factor_cols]
            mean_y = y.mean()
            mean_x = X.mean()
            centered = pd.concat([y - mean_y, X - mean_x], axis=1)
            cov_t = centered.cov()
            sigma_xx = cov_t.loc[factor_cols, factor_cols].to_numpy()
            sigma_xy = cov_t.loc[factor_cols, "pair"].to_numpy()

            try:
                beta_vals = np.linalg.solve(sigma_xx, sigma_xy)
            except np.linalg.LinAlgError:
                beta_vals = np.linalg.pinv(sigma_xx) @ sigma_xy

            beta_calc.loc[t, factor_cols] = beta_vals
            alpha_calc.loc[t] = mean_y - np.dot(mean_x.to_numpy(), beta_vals)

        beta = beta_calc.reindex(pair_ret.index, method="ffill")
        alpha = alpha_calc.reindex(pair_ret.index, method="ffill")
        factors_ret = factors_ret.fillna(0.0)
        fitted = (factors_ret * beta).sum(axis=1)
        residuals = pair_ret - fitted

        return beta, residuals

    def calculate_rolling_pca_loadings(
        self,
        rets,
        window=pd.Timedelta(days=60),
        return_bucket_length=pd.Timedelta(hours=6),
        calc_freq=pd.Timedelta(days=7),
        n_components=2,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

        if isinstance(return_bucket_length, float):
            return_bucket_length = window * return_bucket_length
        if isinstance(calc_freq, float):
            calc_freq = window * calc_freq

        loadings_dict = {f"PC{i+1}": [] for i in range(n_components)}
        explained_var_dict = {f"PC{i+1}": [] for i in range(n_components)}
        index = []

        idx = rets.index
        names = list(rets.keys())

        prev_loadings = None
        last_calc_time = None
        last_checked_friday = None

        for i in range(len(idx)):
            t = idx[i]
            friday_key = t.normalize() if self._is_after_4pm_on_friday(t) else None
            if friday_key is None or friday_key == last_checked_friday:
                continue
            last_checked_friday = friday_key

            should_calculate = (
                last_calc_time is None or (t - last_calc_time) >= calc_freq
            )
            if not should_calculate:
                continue

            rets_until_t = rets.loc[:t]
            agg_rets = (
                rets_until_t.resample(
                    return_bucket_length, label="right", closed="right"
                )
                .sum()
                .dropna(how="all")
            )
            if agg_rets.empty:
                continue

            start_t = t - window - calc_freq
            left = agg_rets.index.searchsorted(start_t, side="left")
            right = agg_rets.index.searchsorted(t, side="right")
            window_data = agg_rets.iloc[left:right]

            # Only run PCA when full window is available
            if window_data.empty or (t - window_data.index[0] < window):
                continue

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
                    if np.dot(current_loadings[comp_idx], prev_loadings[comp_idx]) < 0:
                        current_loadings[comp_idx] = -current_loadings[comp_idx]

            prev_loadings = current_loadings.copy()
            last_calc_time = t

            # Store loadings for each component
            for comp_idx in range(n_components):
                pc_name = f"PC{comp_idx+1}"
                loadings_dict[pc_name].append(current_loadings[comp_idx])
                explained_var_dict[pc_name].append(explained_variance_ratio[comp_idx])

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
        return_bucket_length=pd.Timedelta(hours=6),
        calc_freq=pd.Timedelta(days=7),
        n_components=2,
    ) -> pd.DataFrame:

        usd_rets = rets.filter(regex="^(?!.*UDX).*USD")

        loadings_by_pc, _ = self.calculate_rolling_pca_loadings(
            usd_rets,
            window=window,
            return_bucket_length=return_bucket_length,
            calc_freq=calc_freq,
            n_components=n_components,
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

    def customised_usd_basket(self, rets) -> pd.DataFrame:

        signed_rets = []

        for pair in rets.columns:

            if pair.startswith("USD"):
                signed_rets.append(rets[pair])
            else:
                signed_rets.append(-rets[pair])

        basket = pd.concat(signed_rets, axis=1).mean(axis=1)

        basket.name = "USD_Basket"

        return basket.to_frame()

    def _principal_factor_cache_key(
        self,
        rets: pd.DataFrame,
        method: str,
        window: Union[str, pd.Timedelta],
        return_bucket_length: Union[str, pd.Timedelta],
        calc_freq: Union[str, pd.Timedelta],
        n_components: int,
    ) -> tuple:
        if len(rets.index) == 0:
            first_ts = None
            last_ts = None
        else:
            first_ts = rets.index[0]
            last_ts = rets.index[-1]

        window_key = str(pd.Timedelta(window))
        return_bucket_length_key = str(pd.Timedelta(return_bucket_length))
        calc_freq_key = str(pd.Timedelta(calc_freq))

        return (
            id(self.close_price_dict),
            method,
            window_key,
            return_bucket_length_key,
            calc_freq_key,
            n_components,
            rets.shape,
            tuple(rets.columns),
            first_ts,
            last_ts,
        )

    def compute_principal_factor(
        self,
        rets,
        method="rolling_pca",
        window=pd.Timedelta(days=60),
        return_bucket_length=pd.Timedelta(hours=6),
        calc_freq=pd.Timedelta(days=7),
        n_components=1,
    ) -> pd.DataFrame:
        cache_key = self._principal_factor_cache_key(
            rets,
            method,
            window,
            return_bucket_length,
            calc_freq,
            n_components,
        )
        cached = self.__class__._principal_factor_cache.get(cache_key)
        if cached is not None:
            return cached

        if method == "rolling_pca":
            factor = self.rolling_pca(
                rets,
                window=window,
                return_bucket_length=return_bucket_length,
                calc_freq=calc_freq,
                n_components=n_components,
            )

        elif method == "basket":
            factor = self.customised_usd_basket(rets)
        elif method == "dxy":
            factor = rets["UDXUSD"].to_frame()

        else:
            raise ValueError("Unknown USD factor method")

        self.__class__._principal_factor_cache[cache_key] = factor
        return factor

    def _generate_pair_signal_series(
        self,
        z: pd.Series,
        cum_resid: pd.Series,
        ret_entry_threshold: pd.Series,
    ) -> pd.Series:
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
                if self._is_after_4pm_on_friday(ts):
                    signal.loc[ts] = 0
                    continue
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

        return signal

    def _factor_neutralize_pair_weights(
        self,
        raw_weights: pd.Series,
        beta_matrix: pd.DataFrame,
    ) -> pd.Series:
        weights = raw_weights.astype(float).fillna(0.0)
        betas = beta_matrix.reindex(index=weights.index).fillna(0.0)
        active = weights[weights != 0.0].index

        neutral_weights = pd.Series(0.0, index=weights.index, dtype=float)
        if len(active) == 0:
            return neutral_weights

        active_weights = weights.loc[active]
        active_betas = betas.loc[active]
        B = active_betas.to_numpy(dtype=float)
        s = active_weights.to_numpy(dtype=float)

        if B.size == 0 or np.allclose(B, 0.0):
            neutral_weights.loc[active] = active_weights
            return neutral_weights

        gram = B.T @ B
        factor_projection = B @ (np.linalg.pinv(gram) @ (B.T @ s))
        projected = s - factor_projection

        raw_gross = np.abs(s).sum()
        projected_gross = np.abs(projected).sum()
        if projected_gross > 0 and raw_gross > 0:
            projected = projected * (raw_gross / projected_gross)

        projected[np.abs(projected) < 1e-10] = 0.0
        neutral_weights.loc[active] = projected

        return neutral_weights

    def generate_signals(self) -> dict[str, pd.Series]:

        factor_pca_rolling_lookback = pd.Timedelta(days=60)
        spread_rolling_lookback = pd.Timedelta(minutes=30)

        # Mid returns
        rets = self.compute_mid_returns()

        # USD factor
        factor = self.compute_principal_factor(
            rets,
            method="rolling_pca",
            window=factor_pca_rolling_lookback,
            return_bucket_length=pd.Timedelta(hours=6),
            calc_freq=pd.Timedelta(days=7),
            n_components=1,
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

            signal = self._generate_pair_signal_series(z, cum_resid, ret_entry_threshold)

            signals[pair] = signal

        return signals

    def generate_signals2(self) -> dict[str, pd.Series]:

        factor_pca_rolling_lookback = pd.Timedelta(days=60)
        spread_rolling_lookback = pd.Timedelta(minutes=30)

        # Mid returns
        rets = self.compute_mid_returns()

        # USD factor and loadings
        factor = self.compute_principal_factor(
            rets,
            method="rolling_pca",
            window=factor_pca_rolling_lookback,
            return_bucket_length=pd.Timedelta(hours=6),
            calc_freq=pd.Timedelta(days=7),
            n_components=self.n_components,
        )

        loadings_by_pc, _ = self.calculate_rolling_pca_loadings(
            rets.filter(regex="^(?!.*UDX).*USD"),
            window=factor_pca_rolling_lookback,
            return_bucket_length=pd.Timedelta(hours=12),
            n_components=self.n_components,
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

            signal = self._generate_pair_signal_series(z, cum_resid, ret_entry_threshold)

            signals[pair] = signal

            # Apply hedge in each factor component using its loadings
            for pc_name, pc_loadings in loadings_by_pc.items():
                if pc_loadings.empty:
                    continue
                if pc_name not in beta.columns:
                    continue
                beta_aligned = beta[pc_name].reindex(signal.index)
                scale = (
                    (-signal * beta_aligned)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
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

    def generate_signals3(self) -> dict[str, pd.Series]:

        factor_pca_rolling_lookback = pd.Timedelta(days=60)
        spread_rolling_lookback = pd.Timedelta(minutes=30)
        return_bucket_length = pd.Timedelta(hours=6)
        calc_freq = pd.Timedelta(days=7)

        rets = self.compute_mid_returns()
        factor = self.compute_principal_factor(
            rets,
            method="rolling_pca",
            window=factor_pca_rolling_lookback,
            return_bucket_length=return_bucket_length,
            calc_freq=calc_freq,
            n_components=self.n_components,
        )

        raw_signals: dict[str, pd.Series] = {}
        beta_by_pair: dict[str, pd.DataFrame] = {}

        for pair in self.instruments:
            beta, residuals = self.rolling_regression(
                rets[pair],
                factor,
                window=factor_pca_rolling_lookback,
                return_bucket_length=return_bucket_length,
                calc_freq=calc_freq,
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

            raw_signals[pair] = self._generate_pair_signal_series(
                z, cum_resid, ret_entry_threshold
            ).astype(float)
            beta_by_pair[pair] = beta.reindex(rets.index)

        raw_signal_df = pd.DataFrame(raw_signals, index=rets.index).fillna(0.0)
        signals = {
            pair: pd.Series(0.0, index=rets.index, dtype=float)
            for pair in self.instruments
        }

        for ts in rets.index:
            raw_weights = raw_signal_df.loc[ts, list(self.instruments)]
            beta_matrix = pd.DataFrame(
                {
                    pair: beta_by_pair[pair].loc[ts]
                    for pair in self.instruments
                }
            ).T
            beta_matrix = beta_matrix.reindex(index=list(self.instruments))

            neutral_weights = self._factor_neutralize_pair_weights(
                raw_weights, beta_matrix
            )

            for pair in self.instruments:
                signals[pair].loc[ts] = neutral_weights.loc[pair]

        return signals
