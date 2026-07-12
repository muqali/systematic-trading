from trading.trader import Trader
import pandas as pd
import numpy as np


class AggressiveTrader(Trader):

    def __init__(
        self,
        instruments: list[str],
        close_price_dict: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],
        execute_on_next_price_tick: bool = True,
        vol_window: pd.Timedelta = pd.Timedelta(hours=24),
        vol_median_window: pd.Timedelta = pd.Timedelta(days=7),
        vol_floor: float = 0.5,
        vol_cap: float = 1.5,
    ):
        self.instruments = instruments
        self.close_price_dict = close_price_dict
        self.signals = signals
        self.vol_window = vol_window
        self.vol_median_window = vol_median_window
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self.execute_on_next_price_tick = execute_on_next_price_tick
        self.trade_dict = {}
        self.position_dict = {}
        self.net_pnl_dict = {}
        self.gross_pnl_dict = {}

    def _vol_scale(self, ret: pd.Series) -> pd.Series:
        def _mad(x: np.ndarray) -> float:
            med = np.median(x)
            return 1.4826 * np.median(np.abs(x - med))

        rolling_vol = ret.rolling(window=self.vol_window).apply(_mad, raw=True)
        rolling_median = rolling_vol.rolling(window=self.vol_median_window).median()
        scale = rolling_median / rolling_vol.replace(0, np.nan)
        scale = scale.replace([np.inf, -np.inf], np.nan)
        return scale.clip(lower=self.vol_floor, upper=self.vol_cap)

    def run_signals(self) -> pd.DataFrame:

        for instrument in self.instruments:

            close_price_df = self.close_price_dict.get(instrument)

            sig = self.signals[instrument].reindex(close_price_df.index).fillna(0)

            ret = np.log(close_price_df["mid"]).diff()
            #scale = self._vol_scale(ret).fillna(0.0)
            scale = 1.0
            execution_delay = 1 if self.execute_on_next_price_tick else 0

            position = (sig.shift(execution_delay).fillna(0) * scale).fillna(0.0)

            gross_pnl = position.shift(1) * ret

            trade_size = position.diff()
            trade_mid = close_price_df["mid"]
            trade_price = trade_mid + 0.5 * np.sign(trade_size) * (
                close_price_df["ask"] - close_price_df["bid"]
            )
            trade_cost = trade_size * np.log(trade_price / trade_mid)
            trade = pd.DataFrame(
                {
                    "trade_size": trade_size,
                    "trade_price": trade_price,
                    "trade_cost": trade_cost,
                }
            )
            trade = trade[trade["trade_size"] != 0]

            self.position_dict[instrument] = position
            self.gross_pnl_dict[instrument] = gross_pnl
            self.net_pnl_dict[instrument] = gross_pnl - trade_cost
            self.trade_dict[instrument] = trade

    def generate_positions(self):
        if not self.position_dict:
            self.run_signals()
        return self.position_dict

    def generate_net_pnl(self):
        if not self.net_pnl_dict:
            self.run_signals()
        return self.net_pnl_dict

    def generate_gross_pnl(self):
        if not self.gross_pnl_dict:
            self.run_signals()
        return self.gross_pnl_dict

    def generate_trades(self):
        if not self.trade_dict:
            self.run_signals()
        return self.trade_dict
