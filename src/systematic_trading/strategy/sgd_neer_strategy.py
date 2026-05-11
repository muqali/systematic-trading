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
            hyper_param_dict.get("zscore_window", pd.Timedelta(minutes=120))
        )
        self.zscore_threshold = float(hyper_param_dict.get("zscore_threshold", 2.0))
        if self.zscore_threshold < 0:
            raise ValueError("zscore_threshold must be non-negative.")

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
            past_ret = rets_vs_sgd[ccy].rolling(self.rolling_window).sum()
            rolling_mean = past_ret.rolling(self.zscore_window).mean()
            rolling_std = past_ret.rolling(self.zscore_window).std()
            past_ret_zscore = (past_ret - rolling_mean) / rolling_std

            past_relative_ret = (
                (rets_vs_sgd[ccy] - rets_vs_sgd["index"])
                .rolling(self.rolling_window)
                .sum()
            )
            rolling_mean = past_relative_ret.rolling(self.zscore_window).mean()
            rolling_std = past_relative_ret.rolling(self.zscore_window).std()
            past_relative_ret_zscore = (past_relative_ret - rolling_mean) / rolling_std

            signal = pd.Series(0.0, index=rets_vs_sgd.index, name=ccy)
            signal.loc[
                (index_past_ret_zscore >= self.zscore_threshold) &
                (past_relative_ret_zscore >= 0.0)
            ] = -1.0
            signal.loc[
                (index_past_ret_zscore <= -self.zscore_threshold) &
                (past_relative_ret_zscore <= 0.0)
            ] = 1.0
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
            if ccy in ("USD", "USDSGD"):
                pair_signals.setdefault(
                    "USDSGD", pd.Series(0.0, index=rets_vs_sgd.index, name="USDSGD")
                )
                pair_signals["USDSGD"] = pair_signals["USDSGD"].add(
                    -signal, fill_value=0.0
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
