"""Trend-following systems from Sepp and Lucic (2026).

The strategy implements the European, American, and generalized time-series
momentum (TSMOM) systems described in *The Science and Practice of
Trend-Following Systems* (SSRN 3167787).  Intraday quotes are converted to daily
bars ending at ``trade_entry_time``.  Native ``high``/``low``/``close`` columns
are used when supplied; otherwise OHLC is derived from ``mid``.  The resulting
daily target weights are then held on the original quote index so they can be
consumed directly by :class:`trading.aggressive_trader.AggressiveTrader`.

The returned series are target notional weights, not trade deltas.  A weight
computed at the daily close uses information through that close.  With
``AggressiveTrader``, ``execute_on_next_price_tick=False`` makes that weight earn
the immediately following quote return, matching the paper's timing.  The
trader's existing 17:00 TN carry convention uses the current timestamp's
position.  The American entry weight is held unchanged until exit, matching the
authors' reference implementation and this repository's notional-weight trader
interface.
"""

from collections.abc import Sequence
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from strategy.strategy import Strategy


class TFStrategy(Strategy):
    """European, American, or TSMOM trend-following target weights.

    Parameters are supplied through ``hyper_param_dict``.  The reference
    configuration uses a 250/20-day long-short specification, a 33-day
    volatility/ATR span, and a 260-day annualization factor.  The volatility
    target, American risk multiple, and warm-up are practical configurable
    choices rather than uniquely calibrated empirical values from the paper.

    Common parameters
    -----------------
    tf_style:
        ``"European"``, ``"American"``, or ``"TSMOM"``.
    long_span_days, short_span_days:
        Slow and fast EWMA spans.  For the European system only,
        ``short_span_days=None`` (or ``0``) selects the single-filter version.
    vol_span_days:
        Span of the uncentered EWMA daily-return volatility estimator.
    vol_target:
        Annualized per-instrument volatility target.
    trade_entry_time:
        Eastern-time daily session close used to build bars and schedule
        weights on the quote index.
    warmup_days:
        Number of initial daily bars during which returned weights are zero.

    American-only parameters are ``atr_window_days``,
    ``entry_atr_multiplier``, ``stop_atr_multiplier``, and
    ``risk_multiplier``.  TSMOM-only parameters are ``period_length_days`` and
    ``num_periods``.  Optional ``signal_cap`` and ``weight_cap`` controls are
    disabled by default because they are not part of the paper definitions.
    """

    _STYLE_ALIASES = {
        "european": "European",
        "american": "American",
        "tsmom": "TSMOM",
        "time series momentum": "TSMOM",
        "time_series_momentum": "TSMOM",
        "time-series momentum": "TSMOM",
    }

    def __init__(
        self,
        traded_instruments: Sequence[str],
        fx_price_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict[str, Any] | None = None,
    ):
        params = {} if hyper_param_dict is None else dict(hyper_param_dict)

        style_value = self._first_param(params, "tf_style", "system", default="European")
        style_key = str(style_value).strip().lower()
        if style_key not in self._STYLE_ALIASES:
            valid = "European, American, or TSMOM"
            raise ValueError(f"tf_style must be {valid}.")

        self.tf_style = self._STYLE_ALIASES[style_key]
        self.long_span_days = self._positive_int(
            self._first_param(params, "long_span_days", "long_span", default=250),
            "long_span_days",
        )

        short_span_value = self._first_param(
            params, "short_span_days", "short_span", default=20
        )
        if short_span_value is None or short_span_value == 0:
            self.short_span_days: int | None = None
        else:
            self.short_span_days = self._positive_int(
                short_span_value, "short_span_days"
            )

        if self.tf_style == "American" and self.short_span_days is None:
            raise ValueError("American TF requires a positive short_span_days.")
        if self.short_span_days is not None and (
            self.tf_style in {"European", "American"}
            and self.long_span_days <= self.short_span_days
        ):
            raise ValueError("long_span_days must exceed short_span_days.")

        self.vol_span_days = self._positive_int(
            self._first_param(params, "vol_span_days", "vol_span", default=33),
            "vol_span_days",
        )
        self.annualization_factor = self._positive_float(
            params.get("annualization_factor", 260.0), "annualization_factor"
        )
        self.vol_target = self._positive_float(
            params.get("vol_target", 0.15), "vol_target"
        )
        self.warmup_days = self._nonnegative_int(
            params.get("warmup_days", 250), "warmup_days"
        )

        self.trade_entry_time = params.get("trade_entry_time", "17:00")
        self.calculation_timezone = str(
            params.get("calculation_timezone", "US/Eastern")
        )
        try:
            ZoneInfo(self.calculation_timezone)
        except (TypeError, ZoneInfoNotFoundError) as exc:
            raise ValueError(
                f"Unknown calculation_timezone '{self.calculation_timezone}'."
            ) from exc
        self._entry_offset = self._parse_entry_offset(self.trade_entry_time)

        self.atr_window_days = self._positive_int(
            self._first_param(
                params, "atr_window_days", "atr_window", default=self.vol_span_days
            ),
            "atr_window_days",
        )
        self.entry_atr_multiplier = self._nonnegative_float(
            self._first_param(
                params,
                "entry_atr_multiplier",
                "signal_atr_multiplier",
                default=5.0,
            ),
            "entry_atr_multiplier",
        )
        self.stop_atr_multiplier = self._positive_float(
            self._first_param(
                params,
                "stop_atr_multiplier",
                "stop_loss_atr_multiplier",
                default=5.0,
            ),
            "stop_atr_multiplier",
        )
        self.risk_multiplier = self._positive_float(
            params.get("risk_multiplier", 0.01), "risk_multiplier"
        )

        self.period_length_days = self._positive_int(
            self._first_param(
                params, "period_length_days", "num_ra_returns", default=10
            ),
            "period_length_days",
        )
        self.num_periods = self._positive_int(
            params.get("num_periods", 10), "num_periods"
        )

        self.signal_cap = self._optional_positive_float(
            params.get("signal_cap"), "signal_cap"
        )
        self.weight_cap = self._optional_positive_float(
            self._first_param(
                params, "weight_cap", "weight_abs_limit", default=None
            ),
            "weight_cap",
        )

        self.traded_instruments = tuple(traded_instruments)
        if not self.traded_instruments:
            raise ValueError("traded_instruments must not be empty.")
        if len(set(self.traded_instruments)) != len(self.traded_instruments):
            raise ValueError("traded_instruments must not contain duplicates.")

        self.fx_price_dict = fx_price_dict
        self._validate_price_inputs()

        # Populated by generate_signals for research and diagnostics.
        self.daily_bar_dict: dict[str, pd.DataFrame] = {}
        self.daily_weight_dict: dict[str, pd.Series] = {}
        self.stop_loss_dict: dict[str, pd.Series] = {}

    @staticmethod
    def _first_param(params: dict[str, Any], *names: str, default: Any) -> Any:
        for name in names:
            if name in params:
                return params[name]
        return default

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer.")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer.") from exc
        if converted <= 0 or converted != value:
            raise ValueError(f"{name} must be a positive integer.")
        return converted

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative integer.")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a non-negative integer.") from exc
        if converted < 0 or converted != value:
            raise ValueError(f"{name} must be a non-negative integer.")
        return converted

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be positive.") from exc
        if not np.isfinite(converted) or converted <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return converted

    @staticmethod
    def _nonnegative_float(value: Any, name: str) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be non-negative.") from exc
        if not np.isfinite(converted) or converted < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return converted

    @classmethod
    def _optional_positive_float(cls, value: Any, name: str) -> float | None:
        return None if value is None else cls._positive_float(value, name)

    @staticmethod
    def _parse_entry_offset(value: Any) -> pd.Timedelta:
        if isinstance(value, pd.Timedelta):
            offset = value
        else:
            text = str(value)
            if text.count(":") == 1:
                text = f"{text}:00"
            try:
                offset = pd.to_timedelta(text)
            except (TypeError, ValueError) as exc:
                raise ValueError("trade_entry_time must be a valid time of day.") from exc
        if (
            pd.isna(offset)
            or offset < pd.Timedelta(0)
            or offset >= pd.Timedelta(days=1)
        ):
            raise ValueError("trade_entry_time must fall within one day.")
        return offset

    def _validate_price_inputs(self) -> None:
        for instrument in self.traded_instruments:
            price_df = self.fx_price_dict.get(instrument)
            if price_df is None:
                raise ValueError(f"Missing price data for instrument '{instrument}'.")
            if not isinstance(price_df, pd.DataFrame):
                raise TypeError(f"Price data for '{instrument}' must be a DataFrame.")
            if "mid" not in price_df.columns:
                raise ValueError(f"Expected a 'mid' column for '{instrument}'.")
            if not isinstance(price_df.index, pd.DatetimeIndex):
                raise TypeError(
                    f"Price data for '{instrument}' must use a DatetimeIndex."
                )
            if price_df.empty:
                raise ValueError(f"Price data for '{instrument}' must not be empty.")
            if not price_df.index.is_monotonic_increasing:
                raise ValueError(f"Price index for '{instrument}' must be sorted.")
            if not price_df.index.is_unique:
                raise ValueError(f"Price index for '{instrument}' must be unique.")
            if not pd.api.types.is_numeric_dtype(price_df["mid"]):
                raise TypeError(f"The 'mid' column for '{instrument}' must be numeric.")

            supplied_ohlc = {"high", "low", "close"}.intersection(price_df.columns)
            if supplied_ohlc and supplied_ohlc != {"high", "low", "close"}:
                raise ValueError(
                    f"Price data for '{instrument}' must supply high, low, and "
                    "close together."
                )
            for column in supplied_ohlc:
                if not pd.api.types.is_numeric_dtype(price_df[column]):
                    raise TypeError(
                        f"The '{column}' column for '{instrument}' must be numeric."
                    )
                valid_values = price_df[column].dropna()
                if valid_values.empty:
                    raise ValueError(
                        f"Price data for '{instrument}' has no valid {column} prices."
                    )
                if not np.isfinite(valid_values.to_numpy(dtype=float)).all():
                    raise ValueError(
                        f"{column.capitalize()} prices for '{instrument}' must be finite."
                    )
                if (valid_values <= 0.0).any():
                    raise ValueError(
                        f"{column.capitalize()} prices for '{instrument}' must be positive."
                    )

            if supplied_ohlc:
                high_low = price_df[["high", "low"]].dropna()
                if (high_low["high"] < high_low["low"]).any():
                    raise ValueError(
                        f"High prices for '{instrument}' must not be below low prices."
                    )
                complete_bars = price_df[["high", "low", "close"]].dropna()
                if (
                    (complete_bars["close"] < complete_bars["low"])
                    | (complete_bars["close"] > complete_bars["high"])
                ).any():
                    raise ValueError(
                        f"Close prices for '{instrument}' must lie between high and low."
                    )

            valid_mid = price_df["mid"].dropna()
            if valid_mid.empty:
                raise ValueError(f"Price data for '{instrument}' has no valid mid prices.")
            if not np.isfinite(valid_mid.to_numpy(dtype=float)).all():
                raise ValueError(f"Mid prices for '{instrument}' must be finite.")
            if (valid_mid <= 0.0).any():
                raise ValueError(f"Mid prices for '{instrument}' must be positive.")

    def _to_calculation_timezone(
        self, index: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        index = pd.DatetimeIndex(index)
        try:
            if index.tz is None:
                return index.tz_localize(
                    self.calculation_timezone,
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
            return index.tz_convert(self.calculation_timezone)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not interpret price timestamps in {self.calculation_timezone}."
            ) from exc

    def _daily_bars(self, instrument: str) -> pd.DataFrame:
        price_df = self.fx_price_dict[instrument]
        local_index = self._to_calculation_timezone(price_df.index)
        local_naive = local_index.tz_localize(None)

        # Each observation belongs to the first session close at or after it.
        session_end_naive = (
            (local_naive - self._entry_offset).ceil("1D") + self._entry_offset
        )
        session_end = session_end_naive.tz_localize(
            self.calculation_timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )

        has_ohlc = {"high", "low", "close"}.issubset(price_df.columns)

        def local_series(column: str) -> pd.Series:
            return pd.Series(
                price_df[column].to_numpy(dtype=float),
                index=local_index,
                name=column,
            )

        close = local_series("close" if has_ohlc else "mid")
        high = local_series("high" if has_ohlc else "mid")
        low = local_series("low" if has_ohlc else "mid")
        bars = pd.DataFrame(
            {
                "open": close.groupby(session_end).first(),
                "high": high.groupby(session_end).max(),
                "low": low.groupby(session_end).min(),
                "close": close.groupby(session_end).last(),
            }
        )
        bars = bars.dropna(subset=["close"])
        bars.index.name = "session_end"
        return bars

    def _returns_and_volatility(
        self, bars: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        returns = bars["close"].pct_change(fill_method=None).rename("return")
        variance = returns.pow(2).ewm(
            span=self.vol_span_days,
            adjust=False,
            ignore_na=True,
            min_periods=1,
        ).mean()
        volatility = np.sqrt(variance).mask(lambda value: value <= 0.0)
        volatility.name = "daily_volatility"
        return returns, volatility

    @staticmethod
    def _ewma_from_zero(series: pd.Series, span: int) -> pd.Series:
        raw = series.fillna(0.0).ewm(
            span=span,
            adjust=False,
            ignore_na=False,
        ).mean()
        return raw.where(series.notna())

    @staticmethod
    def _span_decay(span: int) -> float:
        return 1.0 - 2.0 / (span + 1.0)

    def _cap_weights(self, weights: pd.Series) -> pd.Series:
        weights = weights.replace([np.inf, -np.inf], np.nan)
        if self.weight_cap is not None:
            weights = weights.clip(lower=-self.weight_cap, upper=self.weight_cap)
        return weights

    def _apply_warmup(self, weights: pd.Series) -> pd.Series:
        weights = weights.copy()
        if self.warmup_days:
            weights.iloc[: self.warmup_days] = np.nan
        return weights

    def _european_daily_weights(self, bars: pd.DataFrame) -> pd.Series:
        returns, volatility = self._returns_and_volatility(bars)
        vol_normalized_returns = returns.div(volatility.shift(1))

        long_raw = self._ewma_from_zero(
            vol_normalized_returns, self.long_span_days
        )
        long_decay = self._span_decay(self.long_span_days)

        if self.short_span_days is None:
            variance_scale = np.sqrt((1.0 + long_decay) / (1.0 - long_decay))
            signal = variance_scale * long_raw
        else:
            short_raw = self._ewma_from_zero(
                vol_normalized_returns, self.short_span_days
            )
            short_decay = self._span_decay(self.short_span_days)
            q = (
                1.0 / (1.0 - long_decay**2)
                + 1.0 / (1.0 - short_decay**2)
                - 2.0 / (1.0 - long_decay * short_decay)
            ) ** -0.5
            signal = (
                q / (1.0 - long_decay) * long_raw
                - q / (1.0 - short_decay) * short_raw
            )

        if self.signal_cap is not None:
            signal = signal.clip(lower=-self.signal_cap, upper=self.signal_cap)

        volatility_target_weight = self.vol_target / (
            np.sqrt(self.annualization_factor) * volatility
        )
        weights = signal * volatility_target_weight
        return self._cap_weights(weights).rename("European")

    def _average_true_range(self, bars: pd.DataFrame) -> pd.Series:
        previous_close = bars["close"].shift(1)
        true_range = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - previous_close).abs(),
                (bars["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        return true_range.rolling(
            window=self.atr_window_days,
            min_periods=self.atr_window_days,
        ).mean()

    def _american_daily_weights(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        slow = close.ewm(span=self.long_span_days, adjust=False).mean()
        short_span = self.short_span_days
        if short_span is None:  # guarded during construction; narrows the type here
            raise RuntimeError("American TF requires a short EWMA span.")
        fast = close.ewm(span=short_span, adjust=False).mean()
        atr = self._average_true_range(bars)

        weights = np.zeros(len(bars), dtype=float)
        stops = np.full(len(bars), np.nan, dtype=float)
        current_weight = 0.0
        stop = np.nan

        close_values = close.to_numpy(dtype=float)
        slow_values = slow.to_numpy(dtype=float)
        fast_values = fast.to_numpy(dtype=float)
        atr_values = atr.to_numpy(dtype=float)

        for position in range(len(bars)):
            if position < self.warmup_days:
                continue

            price = close_values[position]
            slow_value = slow_values[position]
            fast_value = fast_values[position]
            atr_value = atr_values[position]
            if not np.all(np.isfinite([price, slow_value, fast_value, atr_value])):
                weights[position] = current_weight
                stops[position] = stop
                continue
            if atr_value <= 0.0:
                weights[position] = current_weight
                stops[position] = stop
                continue

            entry_buffer = self.entry_atr_multiplier * atr_value
            long_signal_on = fast_value > slow_value + entry_buffer
            short_signal_on = fast_value < slow_value - entry_buffer

            if current_weight == 0.0:
                relative_size = self.risk_multiplier * price / atr_value
                if self.weight_cap is not None:
                    relative_size = min(relative_size, self.weight_cap)

                if long_signal_on:
                    current_weight = relative_size
                    stop = price - self.stop_atr_multiplier * atr_value
                elif short_signal_on:
                    current_weight = -relative_size
                    stop = price + self.stop_atr_multiplier * atr_value
            elif current_weight > 0.0:
                if price < stop and not long_signal_on:
                    current_weight = 0.0
                    stop = np.nan
                else:
                    stop = max(stop, price - self.stop_atr_multiplier * atr_value)
            else:
                if price > stop and not short_signal_on:
                    current_weight = 0.0
                    stop = np.nan
                else:
                    stop = min(stop, price + self.stop_atr_multiplier * atr_value)

            weights[position] = current_weight
            stops[position] = stop

        weight_series = pd.Series(weights, index=bars.index, name="American")
        self._last_american_stops = pd.Series(
            stops, index=bars.index, name="stop_loss"
        )
        return weight_series

    def _tsmom_daily_weights(
        self,
        bars: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        returns, volatility = self._returns_and_volatility(bars)
        lookback_days = self.period_length_days * self.num_periods
        signal = (
            np.sign(returns)
            .rolling(window=lookback_days, min_periods=lookback_days)
            .sum()
            / np.sqrt(lookback_days)
        )

        # Use a shared portfolio calendar.  The first bar is s_0, so the first
        # L-return rebalance is at calendar index L rather than L - 1.
        rebalance_mask = bars.index.isin(rebalance_dates)
        sparse_weights = pd.Series(np.nan, index=bars.index, dtype=float)
        lagged_annualized_vol = (
            np.sqrt(self.annualization_factor) * volatility.shift(1)
        )
        raw_weights = self.vol_target * signal.div(lagged_annualized_vol)
        sparse_weights.loc[rebalance_mask] = raw_weights.loc[rebalance_mask]

        # The paper forward-fills each rebalance weight to the daily grid.
        weights = sparse_weights.ffill()
        return self._cap_weights(weights).rename("TSMOM")

    def _align_to_price_timezone(
        self, daily_weights: pd.Series, price_index: pd.DatetimeIndex
    ) -> pd.Series:
        daily_weights = daily_weights.copy()
        if price_index.tz is None:
            if daily_weights.index.tz is not None:
                daily_weights.index = daily_weights.index.tz_convert(
                    self.calculation_timezone
                ).tz_localize(None)
        elif daily_weights.index.tz is None:
            daily_weights.index = daily_weights.index.tz_localize(
                self.calculation_timezone
            ).tz_convert(price_index.tz)
        else:
            daily_weights.index = daily_weights.index.tz_convert(price_index.tz)
        return daily_weights

    def _hold_on_price_index(
        self,
        daily_weights: pd.Series,
        price_index: pd.DatetimeIndex,
        instrument: str,
    ) -> pd.Series:
        if daily_weights.empty:
            return pd.Series(0.0, index=price_index, name=instrument)

        daily_weights = self._align_to_price_timezone(daily_weights, price_index)
        combined_index = price_index.union(daily_weights.index).sort_values()
        held = daily_weights.reindex(combined_index).ffill().reindex(price_index)
        held = held.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        held.name = instrument
        return held.astype(float)

    def generate_signals(self) -> dict[str, pd.Series]:
        """Generate target weights on every instrument's exact quote index."""

        signals: dict[str, pd.Series] = {}
        self.daily_bar_dict = {}
        self.daily_weight_dict = {}
        self.stop_loss_dict = {}

        self.daily_bar_dict = {
            instrument: self._daily_bars(instrument)
            for instrument in self.traded_instruments
        }
        common_daily_index = pd.DatetimeIndex([])
        for bars in self.daily_bar_dict.values():
            common_daily_index = common_daily_index.union(bars.index)
        common_daily_index = common_daily_index.sort_values()
        rebalance_dates = common_daily_index[
            self.period_length_days :: self.period_length_days
        ]

        for instrument in self.traded_instruments:
            bars = self.daily_bar_dict[instrument]
            if self.tf_style == "European":
                daily_weights = self._european_daily_weights(bars)
            elif self.tf_style == "American":
                daily_weights = self._american_daily_weights(bars)
                self.stop_loss_dict[instrument] = self._last_american_stops.copy()
            else:
                daily_weights = self._tsmom_daily_weights(
                    bars, rebalance_dates=rebalance_dates
                )

            daily_weights = self._apply_warmup(daily_weights)
            self.daily_weight_dict[instrument] = daily_weights.rename(instrument)

            price_index = pd.DatetimeIndex(self.fx_price_dict[instrument].index)
            signals[instrument] = self._hold_on_price_index(
                daily_weights=daily_weights,
                price_index=price_index,
                instrument=instrument,
            )

        return signals


TrendFollowingStrategy = TFStrategy

__all__ = ["TFStrategy", "TrendFollowingStrategy"]
