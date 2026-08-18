from strategy.strategy import Strategy

import pandas as pd


class IRDMomoStrategy(Strategy):
    def __init__(
        self,
        traded_instruments: tuple[str],
        fx_price_dict: dict[str, pd.DataFrame],
        ccy_yield_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict | None = None,
    ):
        if hyper_param_dict is None:
            hyper_param_dict = {}

        ema_half_life_days = hyper_param_dict.get(
            "ema_half_life_days", hyper_param_dict.get("ema_half_life", 45)
        )
        if ema_half_life_days <= 0:
            raise ValueError("EMA half life must be positive.")

        trade_entry_time = hyper_param_dict.get("trade_entry_time", "11:00")

        tf_style = hyper_param_dict.get("tf_style", "European")
        if tf_style not in {"European", "American"}:
            raise ValueError("tf_style must be either 'European' or 'American'.")

        self.traded_instruments = traded_instruments
        self.fx_price_dict = fx_price_dict
        self.ccy_yield_dict = ccy_yield_dict
        self.hyper_param_dict = hyper_param_dict
        self.ird_dict = self._create_ird_dict()
        self.ccypair_ird_dict = self.ird_dict
        self.ema_half_life_days = ema_half_life_days
        self.trade_entry_time = trade_entry_time
        self.tf_style = tf_style

    @staticmethod
    def _instrument_currencies(instrument: str) -> tuple[str, str]:
        if len(instrument) < 6:
            raise ValueError(
                f"Cannot infer base and term currencies from instrument '{instrument}'."
            )
        return instrument[:3].upper(), instrument[3:6].upper()

    def _yield_series(self, ccy: str) -> pd.Series:
        yield_df = self.ccy_yield_dict.get(ccy)
        if yield_df is None:
            raise ValueError(f"Missing yield data for currency '{ccy}'.")
        if "yield" not in yield_df.columns:
            raise ValueError(f"Expected a 'yield' column for currency '{ccy}'.")
        return yield_df["yield"].sort_index()

    def _create_ird_dict(self) -> dict[str, pd.DataFrame]:
        ird_dict = {}

        for instrument in self.traded_instruments:
            base_ccy, term_ccy = self._instrument_currencies(instrument)
            base_yield = self._yield_series(base_ccy).rename("base_yield")
            term_yield = self._yield_series(term_ccy).rename("term_yield")

            aligned = pd.concat([base_yield, term_yield], axis=1, join="inner")
            ird_dict[instrument] = pd.DataFrame(
                {"ird": aligned["base_yield"] - aligned["term_yield"]},
                index=aligned.index,
            )

        return ird_dict

    def _ird_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {ccypair: ird_df["ird"] for ccypair, ird_df in self.ird_dict.items()}
        ).sort_index()

    def _trade_entry_offset(self) -> pd.Timedelta:
        if isinstance(self.trade_entry_time, pd.Timedelta):
            return self.trade_entry_time

        trade_entry_time = str(self.trade_entry_time)
        if trade_entry_time.count(":") == 1:
            trade_entry_time = f"{trade_entry_time}:00"
        return pd.to_timedelta(trade_entry_time)

    def _trade_entry_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        index = pd.DatetimeIndex(index)
        if index.tz is None:
            index = index.tz_localize("US/Eastern")
        else:
            index = index.tz_convert("US/Eastern")
        return index.normalize() + self._trade_entry_offset()

    def _daily_ird_zscore(self) -> pd.DataFrame:
        ird_panel = self._ird_panel()

        rolling_std = (
            ird_panel.rolling(window=int(self.ema_half_life_days))
            .std()
            .mask(lambda std: std == 0.0)
        )
        return ird_panel.div(rolling_std)

    def _generate_european_signal_panel(self) -> pd.DataFrame:
        daily_zscore = self._daily_ird_zscore()
        ema_zscore = daily_zscore.ewm(
            halflife=self.ema_half_life_days,
            adjust=False,
            ignore_na=True,
        ).mean()
        return ema_zscore.shift(1)

    def _generate_american_signal_panel(self) -> pd.DataFrame:
        ird_panel = self._ird_panel()
        ema_ird = ird_panel.ewm(
            halflife=self.ema_half_life_days,
            adjust=False,
            ignore_na=True,
        ).mean()
        relative_ird = ird_panel - ema_ird
        signal_panel = (relative_ird > 0.0).astype(float).replace({0.0: -1.0})
        signal_panel = signal_panel.mask(relative_ird.isna())
        return signal_panel.shift(1)

    def _align_signal_index_to_price_index(
        self, signal: pd.Series, price_index: pd.DatetimeIndex
    ) -> pd.Series:
        signal = signal.copy()
        if price_index.tz is None and signal.index.tz is not None:
            signal.index = signal.index.tz_convert("US/Eastern").tz_localize(None)
        elif price_index.tz is not None and signal.index.tz is None:
            signal.index = signal.index.tz_localize("US/Eastern").tz_convert(
                price_index.tz
            )
        elif price_index.tz is not None and signal.index.tz is not None:
            signal.index = signal.index.tz_convert(price_index.tz)
        return signal

    def _hold_signal_on_price_index(
        self, signal: pd.Series, price_index: pd.DatetimeIndex
    ) -> pd.Series:
        if signal.empty:
            return pd.Series(0.0, index=price_index, name=signal.name)

        signal = self._align_signal_index_to_price_index(signal, price_index)
        combined_index = price_index.union(signal.index).sort_values()
        held_signal = signal.reindex(combined_index).ffill().reindex(price_index)
        held_signal = held_signal.fillna(0.0)
        held_signal.name = signal.name
        return held_signal

    def _signals_from_panel(self, signal_panel: pd.DataFrame) -> dict[str, pd.Series]:
        signals = {}
        for ccypair in self.traded_instruments:
            signal = signal_panel[ccypair].dropna()
            signal.index = self._trade_entry_index(signal.index)
            signal.name = ccypair
            price_df = self.fx_price_dict.get(ccypair)
            if price_df is not None:
                price_index = pd.DatetimeIndex(price_df.sort_index().index)
                signal = self._hold_signal_on_price_index(signal, price_index)
            signals[ccypair] = signal

        return signals

    def generate_signals(self) -> dict[str, pd.Series]:
        if self.tf_style == "European":
            signal_panel = self._generate_european_signal_panel()
        elif self.tf_style == "American":
            signal_panel = self._generate_american_signal_panel()
        else:
            raise ValueError("tf_style must be either 'European' or 'American'.")

        return self._signals_from_panel(signal_panel)
