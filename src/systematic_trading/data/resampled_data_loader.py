import datetime as dt
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from data.tick_data_loader import _read_tick_quote_file, locate_files as locate_tick_files

RESAMPLED_DATA_DIR = Path.home() / "Programming" / "data" / "resampled"


def _format_histdata_timestamp_index(index: pd.DatetimeIndex) -> pd.Index:
    est_index = index.tz_convert("EST")
    formatted = (
        est_index.strftime("%Y%m%d %H%M%S")
        + est_index.strftime("%f").str.slice(0, 3)
    )
    return pd.Index(formatted, name="timestamp")


def _load_cached_resampled_quote(
    ccy_pair: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    freq: pd.Timedelta,
    output_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    output_dir = RESAMPLED_DATA_DIR if output_dir is None else Path(output_dir)
    freq_code = _resampled_freq_code(freq)
    cache_paths = [
        output_dir / f"DAT_ASCII_{ccy_pair}_{freq_code}_{year}.csv"
        for year in range(start_dt.year, end_dt.year + 1)
    ]

    if not all(path.exists() for path in cache_paths):
        return None

    cached_frames = []
    for path in cache_paths:
        df = _read_tick_quote_file(path)
        if df.empty:
            continue
        cached_frames.append(df)

    if not cached_frames:
        return None

    df = pd.concat(cached_frames)

    mask = (df.index >= start_dt) & (df.index < end_dt)

    return df.loc[mask]


def get_resampled_quote(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    freq: pd.Timedelta = pd.Timedelta("1min"),
    ffill_max_gap: pd.Timedelta = pd.Timedelta("10min"),
) -> pd.DataFrame:
    start_at_first_day_of_year = start_date.month == 1 and start_date.day == 1

    if start_at_first_day_of_year:
        load_start = start_date
    else:
        load_start = start_date - dt.timedelta(days=1)

    if start_at_first_day_of_year:
        start_dt = pd.Timestamp(
            start_date.year, start_date.month, start_date.day, 0, tz="US/Eastern")
    else:
        start_dt = pd.Timestamp(
            start_date.year, start_date.month, start_date.day, 17, tz="US/Eastern"
        ) - pd.Timedelta(days=1)

    end_dt = pd.Timestamp(
        end_date.year, end_date.month, end_date.day, 17, tz="US/Eastern"
    )

    cached_df = _load_cached_resampled_quote(ccy_pair, start_dt, end_dt, freq)
    if cached_df is not None:
        return cached_df

    resampled_frames = []
    file_paths = locate_tick_files(ccy_pair, load_start, end_date)

    for path in file_paths:
        df = _read_tick_quote_file(path)

        resampled_df = df.resample(freq, label="right", closed="right").last()
        resampled_df = resampled_df.loc[
            (resampled_df.index >= start_dt) & (resampled_df.index < end_dt)
        ]
        if resampled_df.empty:
            continue

        resampled_df["mid"] = (resampled_df["bid"] + resampled_df["ask"]) / 2
        resampled_frames.append(resampled_df)

    if not resampled_frames:
        return pd.DataFrame(columns=["bid", "ask", "mid"])

    df = pd.concat(resampled_frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    full = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, tz="US/Eastern"
    )
    trading_mask = (
        (full.dayofweek < 4)
        | ((full.dayofweek == 4) & (full.hour < 17))
        | ((full.dayofweek == 6) & (full.hour >= 18))
    )

    df = df.reindex(full[trading_mask])

    # forward fill with time limit
    limit = int(ffill_max_gap // pd.Timedelta(freq))
    if limit > 0:
        df = df.ffill(limit=limit)

    return df


def get_resampled_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    freq: pd.Timedelta = pd.Timedelta(seconds=15),
    ffill_max_gap: pd.Timedelta = pd.Timedelta("10min"),
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    if len(ccy_pairs) <= 1:
        return {
            ccy_pair: get_resampled_quote(ccy_pair, start_date, end_date, freq, ffill_max_gap)
            for ccy_pair in ccy_pairs
        }

    if max_workers is None:
        max_workers = min(len(ccy_pairs), max(1, min(4, os.cpu_count() or 1)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda ccy_pair: (
                ccy_pair,
                get_resampled_quote(ccy_pair, start_date, end_date, freq, ffill_max_gap),
            ),
            ccy_pairs,
        )
        return {ccy_pair: df for ccy_pair, df in results}


def _resampled_freq_code(freq: pd.Timedelta) -> str:
    total_seconds = int(pd.Timedelta(freq).total_seconds())

    if total_seconds <= 0:
        raise ValueError("freq must be positive.")
    if total_seconds % 3600 == 0:
        return f"H{total_seconds // 3600}"
    if total_seconds % 60 == 0:
        return f"M{total_seconds // 60}"
    return f"S{total_seconds}"


def save_resampled_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    freq: pd.Timedelta = pd.Timedelta(minutes=15),
    output_dir: str | Path = RESAMPLED_DATA_DIR,
) -> dict[str, dict[int, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freq_code = _resampled_freq_code(freq)

    saved_paths = {}

    for ccy_pair in ccy_pairs:
        df = get_resampled_quote(
            ccy_pair=ccy_pair,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
        )
        year_paths = {}

        if not df.empty:
            for year, year_df in df.groupby(df.index.year):
                year_path = output_dir / f"DAT_ASCII_{ccy_pair}_{freq_code}_{year}.csv"
                year_df_to_save = year_df[["bid", "ask"]].copy()
                year_df_to_save.index = _format_histdata_timestamp_index(year_df.index)
                year_df_to_save.to_csv(year_path, header=False)
                year_paths[int(year)] = year_path

        saved_paths[ccy_pair] = year_paths

    return saved_paths
