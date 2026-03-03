"""
data_loader.py
Institutional-grade FX tick → minute → aligned return pipeline
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. TIMEZONE HANDLING
# ============================================================

def ensure_utc_index(df, tz="Etc/GMT+5"):
    """
    Convert fixed EST (UTC-5, no DST) to UTC.
    DO NOT use America/New_York (it has DST).
    """

    if df.index.tz is None:
        df = df.tz_localize(tz)

    return df.tz_convert("UTC")


# ============================================================
# 2. TICK → MID
# ============================================================

def tick_to_mid(df_tick, tz="Etc/GMT+5"):
    """
    Input:
        timestamp | bid | ask
        timestamp assumed fixed EST (UTC-5 no DST)
    Output:
        UTC-indexed dataframe with bid, ask, mid
    """

    df = df_tick.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    df = ensure_utc_index(df, tz=tz)

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]

    return df[["bid", "ask", "mid", "spread"]]


# ============================================================
# 3. MID → MINUTE OHLC
# ============================================================

def mid_to_minute_ohlc(df_mid):
    """
    Convert tick mid to 1-min OHLC.
    Right-labelled, right-closed bars.
    """

    ohlc = df_mid["mid"].resample(
        "1min",
        label="right",
        closed="right"
    ).ohlc()

    spread_1m = df_mid["spread"].resample(
        "1min",
        label="right",
        closed="right"
    ).mean()

    ohlc["spread"] = spread_1m

    return ohlc


def clean_minute_bars(ohlc):
    """
    Handle empty minutes:
    - Forward fill close
    - Set O/H/L = close if missing
    """

    ohlc["close"] = ohlc["close"].ffill()

    mask = ohlc["open"].isna()

    ohlc.loc[mask, "open"] = ohlc.loc[mask, "close"]
    ohlc.loc[mask, "high"] = ohlc.loc[mask, "close"]
    ohlc.loc[mask, "low"]  = ohlc.loc[mask, "close"]

    return ohlc


def tick_to_minute_ohlc(df_tick, tz="Etc/GMT+5"):
    """
    Full tick → minute pipeline.
    """

    df_mid = tick_to_mid(df_tick, tz=tz)
    ohlc = mid_to_minute_ohlc(df_mid)
    ohlc = clean_minute_bars(ohlc)

    return ohlc


# ============================================================
# 4. FX TRADING CALENDAR
# ============================================================

def build_fx_calendar(start, end):
    """
    Institutional FX trading minutes:
    Sunday 22:00 UTC → Friday 22:00 UTC
    """

    full = pd.date_range(
        start=start,
        end=end,
        freq="1min",
        tz="UTC"
    )

    mask = (
        (full.dayofweek < 4) |
        ((full.dayofweek == 4) & (full.hour < 22)) |
        ((full.dayofweek == 6) & (full.hour >= 22))
    )

    return full[mask]


# ============================================================
# 5. MULTI-PAIR ALIGNMENT
# ============================================================

def build_minute_dict(price_tick_dict, tz="Etc/GMT+5"):
    """
    Convert dict of tick dfs → minute OHLC dict.
    """

    minute_dict = {}

    for pair, df_tick in price_tick_dict.items():
        minute_dict[pair] = tick_to_minute_ohlc(df_tick, tz=tz)

    return minute_dict


def extract_close_matrix(minute_dict):
    """
    Build close price matrix from minute dict.
    """

    closes = {
        pair: df["close"]
        for pair, df in minute_dict.items()
    }

    closes = pd.DataFrame(closes).sort_index()
    return closes


def align_to_calendar(closes):
    """
    Align to institutional FX minute calendar.
    """

    calendar = build_fx_calendar(
        start=closes.index.min(),
        end=closes.index.max()
    )

    closes = closes.reindex(calendar)
    return closes


# ============================================================
# 6. MISSING DATA HANDLING
# ============================================================

def limited_forward_fill(closes, max_gap=5):
    """
    Forward fill up to max_gap minutes.
    """

    return closes.ffill(limit=max_gap)


def drop_illiquid_minutes(closes, min_fraction=0.8):
    """
    Drop minutes where too many pairs missing.
    """

    thresh = int(min_fraction * closes.shape[1])
    return closes.dropna(thresh=thresh)


# ============================================================
# 7. RETURN COMPUTATION
# ============================================================

def compute_log_returns(closes):
    """
    Compute log returns.
    Remove session gaps (>1 minute).
    """

    logp = np.log(closes)
    rets = logp.diff()

    time_diff = closes.index.to_series().diff()
    rets[time_diff > pd.Timedelta(minutes=1)] = np.nan

    return rets


# ============================================================
# 8. MASTER PIPELINE
# ============================================================

def prepare_fx_minute_data(price_tick_dict,
                           tz="Etc/GMT+5",
                           max_gap=5,
                           min_fraction=0.8):
    """
    Full institutional FX data pipeline:

    Tick bid/ask
        → UTC mid
        → minute OHLC
        → calendar alignment
        → controlled forward fill
        → liquidity filter
        → log returns
    """

    # Tick → minute OHLC
    minute_dict = build_minute_dict(price_tick_dict, tz=tz)

    # Close matrix
    closes = extract_close_matrix(minute_dict)

    # Align to institutional calendar
    closes = align_to_calendar(closes)

    # Controlled forward fill
    closes = limited_forward_fill(closes, max_gap=max_gap)

    # Remove illiquid minutes
    closes = drop_illiquid_minutes(closes, min_fraction=min_fraction)

    # Compute returns
    rets = compute_log_returns(closes)

    # Drop initial NaNs
    rets = rets.dropna()

    return minute_dict, closes, rets