import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go

from trading.trader import Trader


class Markout:
    def __init__(self, trader: Trader, mid_price_dict: dict[str, pd.DataFrame]):
        self.trades = trader.generate_trades()
        self.mid_price_dict = mid_price_dict

    def calculate_markout(self,
                          plot: bool = False,
                          instruments: list[str] | None = None,
                          start_date: dt.date | None = None,
                          end_date: dt.date | None = None,
                          T_before: pd.Timedelta=pd.Timedelta(minutes=30),
                          T_after: pd.Timedelta=pd.Timedelta(hours=1)) -> pd.DataFrame:

        if instruments is None:
            instruments = list(self.trades.keys())

        if start_date is None or end_date is None:
            trade_indices = [
                trade_df.index
                for instrument, trade_df in self.trades.items()
                if instrument in instruments and trade_df is not None and len(trade_df) > 0
            ]

            if trade_indices:
                all_trade_index = trade_indices[0]
                for idx in trade_indices[1:]:
                    all_trade_index = all_trade_index.union(idx)
                if start_date is None:
                    start_date = all_trade_index.min().date()
                if end_date is None:
                    end_date = all_trade_index.max().date()
            else:
                if start_date is None:
                    start_date = dt.date.min
                if end_date is None:
                    end_date = dt.date.max

        weighted_sum_by_offset: dict[pd.Timedelta, float] = {}
        weight_sum_by_offset: dict[pd.Timedelta, float] = {}

        for instrument in instruments:
            trade_df = self.trades.get(instrument)
            mid_price = self.mid_price_dict.get(instrument)

            if trade_df is None or mid_price is None or len(trade_df) == 0:
                continue

            if isinstance(mid_price, pd.DataFrame):
                if "mid" in mid_price.columns:
                    mid_series = mid_price["mid"]
                elif mid_price.shape[1] == 1:
                    mid_series = mid_price.iloc[:, 0]
                else:
                    raise ValueError(
                        f"mid_price_dict[{instrument!r}] must contain a 'mid' column"
                    )
            else:
                mid_series = mid_price

            trade_mask = (
                (trade_df.index.date >= start_date) & (trade_df.index.date <= end_date)
            )
            filtered_trades = trade_df.loc[trade_mask]

            for trade_time, trade in filtered_trades.iterrows():
                trade_size = trade["trade_size"]
                trade_price = trade["trade_price"]

                if pd.isna(trade_size) or pd.isna(trade_price) or trade_size == 0:
                    continue

                window_mask = (
                    (mid_series.index >= trade_time - T_before)
                    & (mid_series.index <= trade_time + T_after)
                )
                window_mid = mid_series.loc[window_mask].dropna()
                if window_mid.empty:
                    continue

                weight = abs(float(trade_size))
                signed_markout = np.sign(trade_size) * np.log(window_mid / trade_price)
                rel_times = window_mid.index - trade_time

                for offset, markout in zip(rel_times, signed_markout, strict=False):
                    if pd.isna(markout):
                        continue
                    weighted_sum_by_offset[offset] = (
                        weighted_sum_by_offset.get(offset, 0.0) + weight * float(markout)
                    )
                    weight_sum_by_offset[offset] = (
                        weight_sum_by_offset.get(offset, 0.0) + weight
                    )

        if not weight_sum_by_offset:
            return pd.DataFrame(columns=["markout", "weight"])

        offsets = sorted(weight_sum_by_offset.keys())
        result = pd.DataFrame(
            {
                "markout": [
                    weighted_sum_by_offset[offset] / weight_sum_by_offset[offset]
                    for offset in offsets
                ],
                "weight": [weight_sum_by_offset[offset] for offset in offsets],
            },
            index=pd.TimedeltaIndex(offsets, name="time_from_trade"),
        )

        if plot:
            x_minutes = result.index.total_seconds() / 60.0
            y_bps = result["markout"] * 1e4
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_minutes,
                    y=y_bps,
                    mode="lines",
                    name="Markout",
                )
            )
            fig.update_layout(
                title="Trade Markout",
                xaxis_title="Minutes From Trade",
                yaxis_title="Markout (bps)",
                template="plotly_white",
            )
            fig.show()

        return result
