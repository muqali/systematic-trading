from strategy.strategy import Strategy
import numpy as np
import pandas as pd

class SGDNEERStrategy(Strategy):
    SGD = "SGD"
    USDSGD = "USDSGD"

    DEFAULT_WEIGHTS = {
        USDSGD: 0.1987,
        "EURUSD": 0.1503,
        "USDCNH": 0.1476,
        "USDMYR": 0.1162,
        "USDJPY": 0.0938,
        "AUDUSD": 0.065,
        "USDINR": 0.0533,
        "USDKRW": 0.046,
        "USDTHB": 0.1083,
        "USDIDR": 0.0313,
        "USDTWD": 0.0244,
        "GBPUSD": 0.0182,
        "USDHKD": 0.016,
    }

    def __init__(
        self,
        traded_instruments: tuple[str],
        fx_price_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict | None = None,
    ):
        if hyper_param_dict is None:
            hyper_param_dict = {}

        self.rolling_window = pd.Timedelta(
            hyper_param_dict.get("rolling_window", pd.Timedelta(minutes=30))
        )
        self.zscore_window = pd.Timedelta(
            hyper_param_dict.get("zscore_window", pd.Timedelta(days=5))
        )
        self.zscore_entry_threshold = float(
            hyper_param_dict.get("zscore_entry_threshold", 2.0)
        )
        if self.zscore_entry_threshold < 0:
            raise ValueError("zscore_entry_threshold must be non-negative.")

        self.zscore_exit_threshold = float(
            hyper_param_dict.get("zscore_exit_threshold", 0.0)
        )
        if self.zscore_exit_threshold < 0:
            raise ValueError("zscore_exit_threshold must be non-negative.")

        self.spread_rolling_window = pd.Timedelta(
            hyper_param_dict.get("spread_rolling_window", self.rolling_window)
        )
        self.spread_entry_multiplier = float(
            hyper_param_dict.get("spread_entry_multiplier", 5.0)
        )
        if self.spread_entry_multiplier < 0:
            raise ValueError("spread_entry_multiplier must be non-negative.")

        self.weights = dict(hyper_param_dict.get("weights", self.DEFAULT_WEIGHTS))
        self.traded_instruments = tuple(traded_instruments)
        self.fx_price_dict = fx_price_dict

    def compute_mid_returns(self) -> pd.DataFrame:
        mids = {
            pair: df["mid"]
            for pair, df in self.fx_price_dict.items()
            if "mid" in df.columns
        }
        if not mids:
            raise ValueError(
                "fx_price_dict must contain at least one DataFrame with a mid column."
            )

        logp = np.log(pd.DataFrame(mids).sort_index())
        return logp.diff().dropna(how="any")

    @staticmethod
    def _signal_pair_for_ccy(ccy: str) -> str:
        if ccy in ("USD", "USDSGD"):
            return "USDSGD"
        return "USD" + ccy

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

    def _normalised_weights(self, pairs: list[str]) -> dict[str, float]:
        raw_weights = {
            pair: self.weights[pair] for pair in pairs if pair in self.weights
        }
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            raise ValueError("At least one positive NEER weight is required.")
        return {pair: weight / total_weight for pair, weight in raw_weights.items()}

    def _pair_spread_ret(self, pair: str, index: pd.Index) -> pd.Series:
        pair_df = self.fx_price_dict.get(pair)
        if pair_df is None or not {"ask", "bid", "mid"}.issubset(pair_df.columns):
            return pd.Series(np.nan, index=index, dtype=float)

        spread_ret = ((pair_df["ask"] - pair_df["bid"]) / pair_df["mid"]).replace(
            [np.inf, -np.inf], np.nan
        )
        return spread_ret.reindex(index)

    def _ccy_entry_spread_threshold(
        self, ccy: str, index: pd.Index
    ) -> pd.Series:
        if self.spread_entry_multiplier == 0:
            return pd.Series(0.0, index=index, dtype=float)

        if ccy in ("USD", "USDSGD"):
            running_spread = self._pair_spread_ret(self.USDSGD, index)
        else:
            signal_pair, _ = self._pair_signal_for_ccy_signal(
                ccy, pd.Series(0.0, index=index)
            )
            running_spread = self._pair_spread_ret(signal_pair, index)

        return (
            running_spread.rolling(self.spread_rolling_window).mean()
            * self.spread_entry_multiplier
        )

    def _generate_currency_signal_series(
        self,
        entry_zscore: pd.Series,
        direction_zscore: pd.Series,
        past_relative_ret: pd.Series,
        ret_entry_threshold: pd.Series,
    ) -> pd.Series:
        signal = pd.Series(0.0, index=entry_zscore.index)
        position = 0.0

        for ts in entry_zscore.index:
            entry_z = entry_zscore.loc[ts]
            direction_z = direction_zscore.loc[ts]
            rel_ret = past_relative_ret.loc[ts]
            entry_th = ret_entry_threshold.loc[ts]

            if (
                pd.isna(entry_z)
                or pd.isna(direction_z)
                or pd.isna(rel_ret)
                or pd.isna(entry_th)
            ):
                signal.loc[ts] = 0.0
                continue

            if position == 0.0:
                if abs(rel_ret) < entry_th:
                    signal.loc[ts] = 0.0
                    continue
                if (
                    entry_z >= self.zscore_entry_threshold
                    and direction_z >= 0.0
                ):
                    position = -1.0
                elif (
                    entry_z <= -self.zscore_entry_threshold
                    and direction_z <= 0.0
                ):
                    position = 1.0
            elif position == 1.0:
                if entry_z > -self.zscore_exit_threshold:
                    position = 0.0
            elif position == -1.0:
                if entry_z < self.zscore_exit_threshold:
                    position = 0.0

            signal.loc[ts] = position

        return signal

    def build_rets_vs_sgd(self) -> pd.DataFrame:
        rets = self.compute_mid_returns().copy()
        if self.USDSGD not in rets.columns:
            raise ValueError("USDSGD is required to build SGD NEER returns.")

        available_weighted_pairs = [
            pair for pair in self.weights if pair in rets.columns
        ]
        weights = self._normalised_weights(available_weighted_pairs)

        usd_ccy_rets: dict[str, pd.Series] = {"USD": -rets[self.USDSGD]}
        index_ret = pd.Series(0.0, index=rets.index, dtype=float)

        for pair, weight in weights.items():
            if pair == self.USDSGD:
                sgd_ccy_ret = -rets[pair]
            elif pair.startswith("USD"):
                ccy = pair[3:]
                usd_ccy_rets[ccy] = rets[pair]
                sgd_ccy_ret = rets[pair] - rets[self.USDSGD]
            else:
                ccy = pair[:3]
                usd_ccy_rets[ccy] = -rets[pair]
                sgd_ccy_ret = -rets[pair] - rets[self.USDSGD]

            index_ret = index_ret.add(weight * sgd_ccy_ret, fill_value=0.0)

        rets_vs_sgd = pd.DataFrame(index=rets.index)
        rets_vs_sgd["index"] = index_ret
        for ccy in usd_ccy_rets:
            if ccy == "USD":
                rets_vs_sgd[ccy] = usd_ccy_rets[ccy]
            else:
                rets_vs_sgd[ccy] = usd_ccy_rets[ccy] - rets[self.USDSGD]

        return rets_vs_sgd.dropna(how="any")

    def _currency_signals(self, rets_vs_sgd: pd.DataFrame) -> dict[str, pd.Series]:
        signals: dict[str, pd.Series] = {}
        ccy_columns = [col for col in rets_vs_sgd.columns if col != "index"]

        past_index_ret = rets_vs_sgd["index"].rolling(self.rolling_window).sum()
        rolling_mean = past_index_ret.rolling(self.zscore_window).mean()
        rolling_std = past_index_ret.rolling(self.zscore_window).std()
        index_past_ret_zscore = (past_index_ret - rolling_mean) / rolling_std

        for ccy in ccy_columns:
            past_relative_ret = (
                (rets_vs_sgd[ccy] - rets_vs_sgd["index"])
                .rolling(self.rolling_window)
                .sum()
            )
            rolling_mean = past_relative_ret.rolling(self.zscore_window).mean()
            rolling_std = past_relative_ret.rolling(self.zscore_window).std()
            past_relative_ret_zscore = (past_relative_ret - rolling_mean) / rolling_std

            ret_entry_threshold = self._ccy_entry_spread_threshold(
                ccy, rets_vs_sgd.index
            )
            signal = self._generate_currency_signal_series(
                index_past_ret_zscore,
                past_relative_ret_zscore,
                past_relative_ret,
                ret_entry_threshold,
            )
            signal.name = ccy
            signals[ccy] = signal

        return signals

    def generate_signals(self) -> dict[str, pd.Series]:
        rets_vs_sgd = self.build_rets_vs_sgd()
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
                    -signal, fill_value=0.0
                    #0.0, fill_value=0.0
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
