from strategy.strategy import Strategy
from research.sgd_neer import (
    rolling_panel_regression,
    build_rets_vs_sgd,
    DEFAULT_WEIGHTS,
)
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

        self.weights = dict(hyper_param_dict.get("weights", DEFAULT_WEIGHTS))
        self.traded_instruments = tuple(traded_instruments)
        self.fx_price_dict = fx_price_dict


    def _pair_signal_for_ccy_signal(
        self, ccy: str, signal: pd.Series
    ) -> tuple[str, pd.Series]:
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

    def generate_signals(self) -> dict[str, pd.Series]:
        rets_vs_sgd = build_rets_vs_sgd(self.fx_price_dict)
        ccy_signals = self._currency_signals(rets_vs_sgd)
        pair_signals: dict[str, pd.Series] = {
            pair: pd.Series(0.0, index=rets_vs_sgd.index, name=pair)
            for pair in self.traded_instruments
        }

        for ccy, signal in ccy_signals.items():
            if ccy == "USD":
                pair_signals.setdefault(
                    "USDSGD", pd.Series(0.0, index=rets_vs_sgd.index, name="USDSGD")
                )
                pair_signals["USDSGD"] = pair_signals["USDSGD"].add(
                    -signal,
                    fill_value=0.0,
                    # 0.0, fill_value=0.0
                )
                continue

            signal_pair, pair_signal = self._pair_signal_for_ccy_signal(ccy, signal)
            pair_signals.setdefault(
                signal_pair, pd.Series(0.0, index=rets_vs_sgd.index, name=signal_pair)
            )
            pair_signals.setdefault(
                "USDSGD", pd.Series(0.0, index=rets_vs_sgd.index, name="USDSGD")
            )

            pair_signals[signal_pair] = pair_signals[signal_pair].add(
                pair_signal, fill_value=0.0
            )
            pair_signals["USDSGD"] = pair_signals["USDSGD"].add(-signal, fill_value=0.0)

        return pair_signals
