from trading.trader import Trader
import pandas as pd
import numpy as np


class AggressiveTrader(Trader):

    def __init__(
        self,
        instruments: list[str],
        close_price_dict: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],
    ):
        self.instruments = instruments
        self.close_price_dict = close_price_dict
        self.signals = signals
        self.trade_dict = {}
        self.position_dict = {}
        self.net_pnl_dict = {}
        self.gross_pnl_dict = {}

    def run_signals(self) -> pd.DataFrame:

        for instrument in self.instruments:

            close_price_df = self.close_price_dict.get(instrument)

            sig = self.signals[instrument].reindex(close_price_df.index).fillna(0)

            position = sig.shift(1).fillna(0)

            ret = np.log(close_price_df["mid"]).diff()

            gross_pnl = position * ret

            trade_size = position.diff()
            trade_mid = close_price_df["mid"]
            trade_price = trade_mid + 0.5 * trade_size * (
                close_price_df["ask"] - close_price_df["bid"]
            )
            trade_cost = trade_size * (trade_price - trade_mid) / trade_mid
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
