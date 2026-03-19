import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trading.trader import Trader


class PerformanceReport:

    def __init__(
        self,
        instruments: list[str],
        trader: Trader,
    ):
        self.instruments = instruments
        self.net_pnl_dict = trader.generate_net_pnl()
        self.gross_pnl_dict = trader.generate_gross_pnl()
        self.trade_dict = trader.generate_trades()
        self.position_dict = trader.generate_positions()

    def convert_to_daily_pnl(self, pnl_series):
        """Convert to PnL per trade-date"""

        daily_pnl = pnl_series.copy()
        daily_pnl.index = daily_pnl.index + pd.Timedelta(hours=7)
        daily_pnl = daily_pnl.resample("1D").sum()
        return daily_pnl

    def calc_sharpe(self, daily_pnl):
        mean_ret = daily_pnl.mean()
        std_ret = daily_pnl.std()

        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

        return sharpe

    def series_value_duration(self, input_series):
        """
        Calculate duration of each unique non-zero value in a series.
        Can be applied to positions or signals
        """
        s = input_series.copy()
        run_id = s.ne(s.shift()).cumsum()
        runs = s.groupby(run_id).agg(
            signal="first",
            start_time=lambda x: x.index[0],
            end_time=lambda x: x.index[-1],
        )
        next_start = runs["start_time"].shift(-1)
        runs["duration"] = next_start - runs["start_time"]
        runs = runs[runs["signal"].ne(0)]
        dur = runs["duration"].dropna()

        return dur

    def get_instrument_metrics(self, instrument):
        net_pnl = self.net_pnl_dict[instrument]
        gross_pnl = self.gross_pnl_dict[instrument]
        trades = self.trade_dict[instrument]
        position = self.position_dict[instrument]

        metrics = {}

        # ----------------------------------
        # Sharpe
        # ----------------------------------
        daily_net_pnl = self.convert_to_daily_pnl(net_pnl)
        daily_gross_pnl = self.convert_to_daily_pnl(gross_pnl)
        sharpe_net = self.calc_sharpe(daily_net_pnl)
        sharpe_gross = self.calc_sharpe(daily_gross_pnl)

        # ----------------------------------
        # Worst day
        # ----------------------------------
        worst_day_pnl = daily_net_pnl.min()

        # ----------------------------------
        # Max drawdown
        # ----------------------------------
        cumulative = net_pnl.cumsum()
        peak = cumulative.cummax()
        drawdown = cumulative - peak
        max_dd = drawdown.min()

        # ----------------------------------
        # Avg trades per day
        # ----------------------------------
        daily_trades = trades.copy()
        daily_trades.index = daily_trades.index + pd.Timedelta(hours=7)
        daily_trades = trades["trade_size"].resample("1D").count()
        avg_trades = daily_trades.mean()

        # ----------------------------------
        # Avg PnL per trade
        # ----------------------------------
        total_trades = len(trades)

        avg_net_pnl_per_trade = net_pnl.sum() / total_trades if total_trades > 0 else 0
        avg_gross_pnl_per_trade = (
            gross_pnl.sum() / total_trades if total_trades > 0 else 0
        )

        # ----------------------------------
        # Position duration
        # ----------------------------------
        position_duration = self.series_value_duration(position)
        avg_position_duration = (
            position_duration.mean() if len(position_duration) > 0 else 0
        )
        median_position_duration = (
            position_duration.median() if len(position_duration) > 0 else 0
        )
        g10_position_duration = (
            position_duration.quantile(0.1) if len(position_duration) > 0 else 0
        )
        g90_position_duration = (
            position_duration.quantile(0.9) if len(position_duration) > 0 else 0
        )

        # ----------------------------------
        # Store (same structure as before)
        # ----------------------------------
        metrics["Sharpe_Net"] = sharpe_net
        metrics["Sharpe_Gross"] = sharpe_gross
        metrics["AvgTradesPerDay"] = avg_trades
        metrics["MaxDrawdown"] = max_dd
        metrics["WorstDayPnL"] = worst_day_pnl
        metrics["AvgNetPnLPerTrade"] = avg_net_pnl_per_trade
        metrics["AvgGrossPnLPerTrade"] = avg_gross_pnl_per_trade
        metrics["AvgPosDur"] = avg_position_duration
        metrics["MedPosDur"] = median_position_duration
        metrics["Q10PosDur"] = g10_position_duration
        metrics["Q90PosDur"] = g90_position_duration

        return metrics

    def generate_performance_report(self):

        report_dict = {}

        # --------------------------------------------------
        # 1️⃣ Per Currency Pair Metrics
        # --------------------------------------------------

        for instrument in self.instruments:
            report_dict[instrument] = self.get_instrument_metrics(instrument)

        report_df = pd.DataFrame(report_dict).T

        return report_df

    def plot_cumulative_pnl(
        self,
        instrument: str,
        ax=None,
        title: str | None = None,
    ):
        net_pnl = self.net_pnl_dict[instrument]
        gross_pnl = self.gross_pnl_dict[instrument]
        cumulative_net = net_pnl.cumsum()
        cumulative_gross = gross_pnl.cumsum()

        if ax is None:
            _, ax = plt.subplots()

        cumulative_net.plot(ax=ax, label="Net")
        cumulative_gross.plot(ax=ax, label="Gross")
        plot_title = title or f"{instrument} Cumulative PnL"
        ax.set_title(plot_title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax
