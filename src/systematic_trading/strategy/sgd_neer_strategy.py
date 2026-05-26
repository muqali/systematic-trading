from strategy.strategy import Strategy
from research.sgd_neer import (
    rolling_panel_regression,
    build_rets_vs_sgd,
    DEFAULT_WEIGHTS,
)
from util.math_util import mad_clip
import numpy as np
import pandas as pd


class SGDNEERStrategy(Strategy):
    SGD = "SGD"
    USDSGD = "USDSGD"

    def __init__(
        self,
        traded_instruments: tuple[str],
        fx_price_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict | None = None,
    ):
        if hyper_param_dict is None:
            hyper_param_dict = {}

        self.past_horizon = pd.Timedelta(
            hyper_param_dict.get("past_horizon", pd.Timedelta(minutes=120))
        )
        self.zscore_window = pd.Timedelta(
            hyper_param_dict.get("zscore_window", pd.Timedelta(days=5))
        )
        self.future_horizon = pd.Timedelta(
            hyper_param_dict.get("future_horizon", pd.Timedelta(minutes=15))
        )
        self.first_stage_beta_mode = hyper_param_dict.get(
            "first_stage_beta_mode", "regression"
        )
        self.zscore_entry_threshold = float(
            hyper_param_dict.get("zscore_entry_threshold", 2.0)
        )
        if self.zscore_entry_threshold < 0:
            raise ValueError("zscore_entry_threshold must be non-negative.")

        self.spread_rolling_window = pd.Timedelta(
            hyper_param_dict.get("spread_rolling_window", self.past_horizon)
        )
        self.margin_multiple = float(hyper_param_dict.get("margin_multiple", 2.0))
        if self.margin_multiple < 0:
            raise ValueError("margin_multiple must be non-negative.")
        self.adjust_to_target = bool(hyper_param_dict.get("adjust_to_target", True))
        self.delay_trade_on_signal = bool(
            hyper_param_dict.get("delay_trade_on_signal", False)
        )
        self.calibration_lookback_weeks = int(
            hyper_param_dict.get("calibration_lookback_weeks", 12)
        )
        self.calibration_freq = hyper_param_dict.get("calibration_freq", "W-SAT")

        self.weights = dict(hyper_param_dict.get("weights", DEFAULT_WEIGHTS))
        self.traded_instruments = tuple(traded_instruments)
        self.fx_price_dict = fx_price_dict

    def _pair_signal_for_ccy_signal(
        self, ccy: str, signal: pd.Series
    ) -> tuple[str, pd.Series]:
        if ccy == "USD":
            return self.USDSGD, -signal

        usd_ccy_pair = "USD" + ccy
        ccy_usd_pair = ccy + "USD"

        if usd_ccy_pair in self.traded_instruments:
            return usd_ccy_pair, signal
        if ccy_usd_pair in self.traded_instruments:
            return ccy_usd_pair, -signal
        return usd_ccy_pair, signal

    def _pair_spread_ret(self, pair: str, index: pd.Index) -> pd.Series:
        pair_df = self.fx_price_dict.get(pair)
        if pair_df is None or not {"ask", "bid", "mid"}.issubset(pair_df.columns):
            return pd.Series(np.nan, index=index, dtype=float)

        spread_ret = ((pair_df["ask"] - pair_df["bid"]) / pair_df["mid"]).replace(
            [np.inf, -np.inf], np.nan
        )
        return spread_ret.reindex(index)

    def _price_pair_for_ccy(self, ccy: str) -> str:
        if ccy in ("USD", "USDSGD"):
            return self.USDSGD

        usd_ccy_pair = "USD" + ccy
        ccy_usd_pair = ccy + "USD"

        if usd_ccy_pair in self.fx_price_dict:
            return usd_ccy_pair
        if ccy_usd_pair in self.fx_price_dict:
            return ccy_usd_pair
        return self._pair_signal_for_ccy_signal(ccy, pd.Series(dtype=float))[0]

    def _ccy_round_trip_spread_ret(self, ccy: str, index: pd.Index) -> pd.Series:
        if ccy in ("USD", "USDSGD"):
            return self._pair_spread_ret(self.USDSGD, index)

        ccy_leg_pair = self._price_pair_for_ccy(ccy)
        ccy_leg_spread = self._pair_spread_ret(ccy_leg_pair, index)
        sgd_leg_spread = self._pair_spread_ret(self.USDSGD, index)
        return ccy_leg_spread + sgd_leg_spread

    def _rolling_half_spread_tcost(self, ccy: str, index: pd.Index) -> pd.Series:
        spread = self._ccy_round_trip_spread_ret(ccy, index)
        return 0.5 * spread.rolling(self.spread_rolling_window).mean()

    def _index_full_spread_cost(self) -> pd.Series:
        price = self.fx_price_dict.get(self.USDSGD)
        spread = (price["ask"] - price["bid"]) / price["mid"]

        for pair in self.traded_instruments:
            if pair == self.USDSGD:
                continue
            else:
                price = self.fx_price_dict.get(pair)
                spread += (
                    self.weights.get(pair)
                    * (price["ask"] - price["bid"])
                    / price["mid"]
                )

        return spread

    def _throttle_prediction_signal(
        self, prediction_df: pd.DataFrame, tcost: pd.Series
    ) -> pd.Series:
        if prediction_df.empty:
            return pd.Series(dtype=float)

        df = prediction_df.join(tcost.rename("tcost"), how="left")
        delay = 1 if self.delay_trade_on_signal else 0
        target_position = (
            df["prediction"]
            .shift(delay)
            .where(
                df["prediction_zscore"].shift(delay).abs()
                > self.zscore_entry_threshold,
                0.0,
            )
        )

        margin = self.margin_multiple * df["tcost"]
        positions = np.empty(len(df))
        positions[:] = np.nan

        target = target_position.to_numpy(dtype=float)
        margin_vals = margin.to_numpy(dtype=float)

        valid = np.flatnonzero(np.isfinite(target) & np.isfinite(margin_vals))
        if len(valid) == 0:
            return pd.Series(0.0, index=df.index, dtype=float)

        start = valid[0]
        positions[:start] = 0.0
        positions[start] = target[start]

        for i in range(start + 1, len(df)):
            current_pos = positions[i - 1]
            t = target[i]
            m = margin_vals[i]

            if not np.isfinite(t) or not np.isfinite(m):
                positions[i] = current_pos
                continue

            lower = t - m
            upper = t + m

            if current_pos < lower:
                positions[i] = t if self.adjust_to_target else lower
            elif current_pos > upper:
                positions[i] = t if self.adjust_to_target else upper
            else:
                positions[i] = current_pos

        return pd.Series(positions, index=df.index, dtype=float)

    def _currency_signals(self, rets_vs_sgd: pd.DataFrame) -> dict[str, pd.Series]:
        signals: dict[str, pd.Series] = {}
        ccy_columns = [col for col in rets_vs_sgd.columns if col != "index"]

        for ccy in ccy_columns:
            prediction_df = rolling_panel_regression(
                rets_vs_sgd["index"],
                rets_vs_sgd[ccy],
                past_horizon=self.past_horizon,
                future_horizon=self.future_horizon,
                lookback_window=self.zscore_window,
                first_stage_beta_mode=self.first_stage_beta_mode,
                pair_weights=self.weights,
            )
            tcost = self._rolling_half_spread_tcost(ccy, prediction_df.index)
            signal = self._throttle_prediction_signal(
                prediction_df,
                tcost,
            )
            signal = signal.reindex(rets_vs_sgd.index).fillna(0.0)
            signal.name = ccy
            signals[ccy] = signal

        return signals

    def calibrate_return_regression_stats(
        self,
        ret_series: pd.Series | None = None,
        lookback_weeks: int = 12,
        calibration_freq: str = "W-SAT",
        past_horizon: pd.Timedelta | None = None,
        future_horizon: pd.Timedelta | None = None,
        winsor_z: float = 3.0,
    ) -> pd.DataFrame:
        """Periodically fit future returns on past returns.

        By default this calibrates every Saturday using observations from the
        previous ``lookback_weeks``. Past and future returns are winsorised
        inside each calibration sample before regression. If ``ret_series`` is
        not supplied, the SGD NEER index return series is used.
        """
        if lookback_weeks <= 0:
            raise ValueError("lookback_weeks must be positive.")

        if ret_series is None:
            ret_series = build_rets_vs_sgd(self.fx_price_dict)["index"]
        if not isinstance(ret_series.index, pd.DatetimeIndex):
            raise TypeError("ret_series must have a DatetimeIndex.")

        ret_series = ret_series.sort_index().dropna()
        if ret_series.empty:
            return pd.DataFrame().rename_axis("calibration_ts")
        if len(ret_series.index) < 2:
            return pd.DataFrame().rename_axis("calibration_ts")

        past_horizon = self.past_horizon if past_horizon is None else past_horizon
        future_horizon = (
            self.future_horizon if future_horizon is None else future_horizon
        )
        freq = ret_series.index.to_series().diff().dropna().min()
        if pd.isna(freq) or freq <= pd.Timedelta(0):
            raise ValueError("ret_series index must be strictly increasing.")
        p = int(past_horizon / freq)
        f = int(future_horizon / freq)
        if p < 1 or f < 1:
            raise ValueError(
                "past_horizon and future_horizon must be at least one period."
            )

        return_past = ret_series.rolling(window=p).sum()
        return_future = ret_series.rolling(window=f).sum().shift(-f)
        regression_sample = pd.DataFrame(
            {
                "return_past": return_past,
                "return_future": return_future,
            }
        ).dropna()

        lookback_window = pd.Timedelta(weeks=lookback_weeks)
        calibration_dates = pd.date_range(
            start=ret_series.index.min().normalize(),
            end=ret_series.index.max() + pd.Timedelta(days=1),
            freq=calibration_freq,
        )

        rows = []
        first_calibration_ts = (
            ret_series.index.min() + lookback_window + future_horizon
        )
        for calibration_ts in calibration_dates:
            if calibration_ts < first_calibration_ts:
                continue

            sample_start = calibration_ts - lookback_window
            sample_end = calibration_ts - future_horizon
            sample = regression_sample.loc[
                (regression_sample.index >= sample_start)
                & (regression_sample.index <= sample_end)
            ]
            sample = mad_clip(sample, z=winsor_z).dropna()

            row = {
                "sample_start": sample_start,
                "sample_end": sample_end,
                "past_horizon": past_horizon,
                "future_horizon": future_horizon,
                "lookback_weeks": lookback_weeks,
                "winsor_z": winsor_z,
                "n_observations": len(sample),
                "alpha": np.nan,
                "beta": np.nan,
                "correlation": np.nan,
                "r_squared": np.nan,
                "std_err": np.nan,
                "t_stat": np.nan,
            }

            x = sample["return_past"]
            y = sample["return_future"]
            x_var = x.var()
            y_var = y.var()
            if len(sample) >= 3 and x_var > 0 and y_var > 0:
                beta = y.cov(x) / x_var
                alpha = y.mean() - beta * x.mean()
                residual = y - alpha - beta * x
                denom = ((x - x.mean()) ** 2).sum()
                s2 = (residual**2).sum() / (len(sample) - 2)
                std_err = np.sqrt(s2 / denom)
                correlation = x.corr(y)

                row.update(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "correlation": correlation,
                        "r_squared": correlation**2,
                        "std_err": std_err,
                        "t_stat": beta / std_err if std_err > 0 else np.nan,
                    }
                )

            rows.append((calibration_ts, row))

        if not rows:
            return pd.DataFrame().rename_axis("calibration_ts")

        result = pd.DataFrame(
            [row for _, row in rows],
            index=pd.DatetimeIndex([ts for ts, _ in rows], name="calibration_ts"),
        )
        result.attrs["ret_series_name"] = ret_series.name
        result.attrs["calibration_freq"] = calibration_freq
        return result

    def _index_signal(self, index_ret_series: pd.Series) -> pd.Series:
        freq = index_ret_series.index.to_series().diff().dropna().min()
        if pd.isna(freq) or freq <= pd.Timedelta(0):
            return pd.Series(0.0, index=index_ret_series.index, name="index")

        past_window = int(self.past_horizon / freq)
        zscore_window = int(self.zscore_window / freq)
        if past_window < 1 or zscore_window < 2:
            return pd.Series(0.0, index=index_ret_series.index, name="index")

        calibration_stats = self.calibrate_return_regression_stats(
            index_ret_series,
            lookback_weeks=self.calibration_lookback_weeks,
            calibration_freq=self.calibration_freq,
            past_horizon=self.past_horizon,
            future_horizon=self.future_horizon,
        )
        if calibration_stats.empty:
            return pd.Series(0.0, index=index_ret_series.index, name="index")

        return_past = index_ret_series.rolling(window=past_window).sum()
        beta = calibration_stats["beta"].reindex(index_ret_series.index, method="ffill")

        # validate beta: mean reversion should mean negative beta
        beta = beta.clip(upper=0.0)

        prediction = beta * return_past
        roll_prediction = prediction.rolling(window=zscore_window)
        prediction_zscore = (
            (prediction - roll_prediction.mean()) / roll_prediction.std()
        )
        prediction_df = pd.DataFrame(
            {
                "return_past": return_past,
                "calibration_beta": beta,
                "prediction": prediction,
                "prediction_zscore": prediction_zscore,
            }
        ).dropna()

        tcost = self._index_full_spread_cost()
        signal = self._throttle_prediction_signal(prediction_df, tcost)
        signal = signal.reindex(index_ret_series.index).fillna(0.0)
        signal.name = "index"

        return signal

    def _index_signal_to_pair_signal(
        self, index_signal: pd.Series, ccys: list[str]
    ) -> dict[str, pd.Series]:
        pair_signals: dict[str, pd.Series] = {
            pair: pd.Series(0.0, index=index_signal.index, name=pair)
            for pair in self.traded_instruments
        }

        for ccy in ccys:
            signal_pair, pair_signal = self._pair_signal_for_ccy_signal(
                ccy, index_signal
            )
            weight = self.weights.get(signal_pair)
            pair_signals[signal_pair] = pair_signal * weight

            if ccy != "USD":
                pair_signals[self.USDSGD] = -index_signal * weight

        return pair_signals

    def generate_signals(self) -> dict[str, pd.Series]:
        rets_vs_sgd = build_rets_vs_sgd(self.fx_price_dict).dropna(how="any")

        index_signal = self._index_signal(rets_vs_sgd["index"])

        pair_signals = self._index_signal_to_pair_signal(
            index_signal, rets_vs_sgd.columns.drop("index")
        )

        return pair_signals
