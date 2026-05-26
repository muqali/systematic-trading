import numpy as np
import pandas as pd

def convert_to_daily_pnl(pnl_series):
    """Convert to PnL per trade-date"""

    daily_pnl = pnl_series.copy()
    daily_pnl.index = daily_pnl.index + pd.Timedelta(hours=7)
    daily_pnl = daily_pnl.resample("1D").sum()
    return daily_pnl

def calc_sharpe(pnl_series):
    daily_pnl = convert_to_daily_pnl(pnl_series)
    mean_ret = daily_pnl.mean()
    std_ret = daily_pnl.std()

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

    return sharpe