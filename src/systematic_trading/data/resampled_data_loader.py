import datetime as dt
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pandas as pd

from data.tick_data_loader import (
    ANNUAL_TICK_CHUNK_SIZE,
    DEFAULT_SOURCE,
    LONDON_SYNCED_SOURCE,
    _iter_tick_quote_file,
    _iter_processed_tick_file,
    _read_quote_file,
    get_raw_source_settings,
    quote_input_files,
)

RESAMPLED_DATA_DIR = Path.home() / "Programming" / "data" / "resampled"
CACHE_TIMEZONE = "US/Eastern"


def _trading_mask(index: pd.DatetimeIndex):
    """The existing FX-session convention; holidays are not inferred."""
    # Extract local wall time once. Each field access on a timezone-aware
    # index otherwise repeats the expensive UTC-to-local conversion.
    local = index.tz_localize(None) if index.tz is not None else index
    weekday, hour = local.dayofweek, local.hour
    return (
        (weekday < 4)
        | ((weekday == 4) & (hour < 17))
        | ((weekday == 6) & (hour >= 18))
    )


def _trading_index(start, end, freq):
    first = start.tz_convert("UTC").ceil(freq).tz_convert(CACHE_TIMEZONE)
    index = pd.date_range(first, end, freq=freq, inclusive="left")
    return index[_trading_mask(index)]


def _trading_bounds(start, end, freq):
    """First/last requested bars, without allocating a multi-year grid.

    The session closes only over weekends, so three elapsed days at either
    edge suffice, including DST weekends. Callers use hour-dividing grids.
    """
    padding = pd.Timedelta(days=3)
    first = _trading_index(start, min(end, start + padding), freq)
    if first.empty:
        return None
    last = first if end <= start + padding else _trading_index(max(start, end - padding), end, freq)
    return first[0], last[-1]


def _on_trading_grid(index, freq):
    """Validate session membership and UTC alignment without building a grid."""
    return bool(
        not index.hasnans
        and (index.as_unit("ns").asi8 % freq.value == 0).all()
        and _trading_mask(index).all()
    )


def _cache_covers_window(frame, start, end, freq, *, current_year=None):
    """Trust completed-year files; only check the tail of an ongoing year.

    frame must contain the whole annual cache, before request slicing. Sparse
    prices/timestamps, including holiday edges, do not prove incompleteness.
    A current/future-year file is reusable when it reaches the last requested
    trading bar; its first timestamp and internal gaps are not coverage tests.
    """
    if frame is None or frame.empty:
        return False
    if current_year is None:
        current_year = pd.Timestamp.now(tz=CACHE_TIMEZONE).year
    if start.year < current_year:
        return True
    bounds = _trading_bounds(start, end, freq)
    return bounds is None or frame.index[-1] >= bounds[1]


def _resampled_request_bounds(start_date, end_date):
    """Shared half-open Eastern window, preserving the historical date API."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date.")
    if start_date.month == 1 and start_date.day == 1:
        start = pd.Timestamp(start_date.year, start_date.month, start_date.day, tz=CACHE_TIMEZONE)
    else:
        start = pd.Timestamp(
            start_date.year, start_date.month, start_date.day, 17, tz=CACHE_TIMEZONE
        ) - pd.Timedelta(days=1)
    end = pd.Timestamp(end_date.year, end_date.month, end_date.day, 17, tz=CACHE_TIMEZONE)
    return start, end


def _format_histdata_timestamp_index(
    index: pd.DatetimeIndex,
    timezone: str = CACHE_TIMEZONE,
) -> pd.Index:
    output_index = index.tz_convert(timezone)
    formatted = (
        output_index.strftime("%Y%m%d %H%M%S")
        + output_index.strftime("%f").str.slice(0, 3)
    )
    return pd.Index(formatted, name="timestamp")


def _load_cached_resampled_quote(
    ccy_pair: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    freq: pd.Timedelta,
    cache_dir: str | Path | None = None,
    cache_timezone: str = CACHE_TIMEZONE,
    source: str | None = DEFAULT_SOURCE,
) -> pd.DataFrame | None:
    """Read selected-source rows; coverage is checked separately by the caller.

    None means a missing/unreadable cache. A partial DataFrame is retained so
    it need not be discarded if the source ticks are unavailable as well.
    """
    cache_dir = RESAMPLED_DATA_DIR if cache_dir is None else Path(cache_dir)
    source = DEFAULT_SOURCE if source is None else source
    cache_pair = get_raw_source_settings(source).format_cache_pair(ccy_pair)
    freq_code = _resampled_freq_code(freq)
    # end_dt is exclusive: midnight on Jan 1 does not require that year's file.
    years = range(start_dt.year, (end_dt - pd.Timedelta(1, "ns")).year + 1)
    cache_paths = [
        cache_dir / f"{source}_{cache_pair}_{freq_code}_{year}.csv"
        for year in years
    ]

    if not all(path.is_file() for path in cache_paths):
        return None

    cached_frames = []
    for year, path in zip(years, cache_paths):
        try:
            df = _read_quote_file(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
            return None
        if df.empty:
            continue
        df.index = df.index.tz_localize(
            cache_timezone, ambiguous="NaT", nonexistent="NaT"
        ).tz_convert(CACHE_TIMEZONE)
        if (
            df.index.hasnans or not df.index.is_unique
            or not df.index.is_monotonic_increasing
            or not (df.index.year == year).all()
        ):
            return None
        df["mid"] = (df["bid"] + df["ask"]) / 2
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
    *,
    cache_dir: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    tick_data_dir: str | Path | None = None,
    prepared_raw_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
    source_timezone: str | None = None,
    timestamp_format: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Reuse annual caches; resample missing/invalid or outdated years from ticks.

    Only registered provider prefixes (HISTDATA/DUKASCOPY) are cache sources.
    tick_data_dir contains normalized Eastern ticks; raw_data_dir overrides
    Documents/data/<source>. prepared_raw_dir is reserved for temporary CSVs
    already extracted by the exporter. Raw clock rules never apply to caches.
    The compact, headerless Eastern resampled output format is unchanged.
    Valid cache files for calendar years before the current US/Eastern year
    are assumed complete, even if arbitrarily sparse or missing holiday edges.
    Current/future-year caches are reused if their last timestamp reaches the
    last requested trading bar. Neither first timestamps nor internal gaps
    trigger rebuilding. NaN prices remain valid cache entries. Missing,
    unreadable, unordered, duplicate, wrong-year or off-grid files still rebuild.
    Intentionally partial historical files are trusted too; regenerate them
    with the exporter when more source data arrives, or bypass with use_cache=False.
    Consecutive missing years share one tick pass, including a small preceding
    context for right-closed bars and forward filling. Cached prices are never
    filled a second time. Non-hour-dividing frequencies use the full tick path.
    Reads never save or overwrite caches. use_cache=False resamples directly.
    Raw fallback selects only files whose source calendar can contribute to
    the requested trading bars, including preceding fill/bar context.
    """
    freq = pd.Timedelta(freq)
    ffill_max_gap = pd.Timedelta(ffill_max_gap)
    if (
        pd.isna(freq) or pd.isna(ffill_max_gap)
        or freq <= pd.Timedelta(0) or ffill_max_gap < pd.Timedelta(0)
    ):
        raise ValueError("freq must be positive and ffill_max_gap nonnegative.")
    start_dt, end_dt = _resampled_request_bounds(start_date, end_date)
    get_raw_source_settings(source)  # Validate even for a closed-session request.

    options = dict(
        raw_data_dir=raw_data_dir, tick_data_dir=tick_data_dir,
        prepared_raw_dir=prepared_raw_dir, source=source,
        source_timezone=source_timezone, timestamp_format=timestamp_format,
    )
    # Non-hour-dividing grids can change phase at local day/year/DST boundaries.
    # Keep the full-range resampling path for them rather than join shifted bars.
    if not use_cache or pd.Timedelta(hours=1) % freq:
        return _resample_from_ticks(
            ccy_pair, start_dt, end_dt, freq, ffill_max_gap, **options
        )

    missing_years = []
    results = []
    for year in range(start_dt.year, end_dt.year + 1):
        year_start = pd.Timestamp(year, 1, 1, tz=CACHE_TIMEZONE)
        year_end = pd.Timestamp(year + 1, 1, 1, tz=CACHE_TIMEZONE)
        start = max(start_dt, year_start)
        end = min(end_dt, year_end)
        if _trading_bounds(start, end, freq) is None:
            continue
        cached = _load_cached_resampled_quote(
            ccy_pair, year_start, year_end, freq,
            cache_dir=cache_dir, source=source,
        )
        if cached is not None and not _on_trading_grid(cached.index, freq):
            # An off-grid cache is invalid, not a usable partial cache.
            cached = None
        covered = _cache_covers_window(cached, start, end, freq)
        if cached is not None:
            cached = cached.loc[(cached.index >= start) & (cached.index < end)]
        if covered:
            if not cached.empty:
                results.append(cached)
        else:
            missing_years.append((start, end, cached))

    # Rebuild consecutive missing/invalid/outdated years together, preserving
    # bars/fills between them without repeatedly extracting the same source ZIP.
    spans = []
    for start, end, cached in missing_years:
        if spans and spans[-1][1] == start:
            spans[-1][1] = end
            spans[-1][2].append(cached)
        else:
            spans.append([start, end, [cached]])
    for start, end, partial_caches in spans:
        # Read enough preceding trading bars to preserve the remaining fill
        # budget. Cached values have already been filled, so cannot seed this
        # accurately; in particular, never ffill concatenated caches again.
        padding = max(ffill_max_gap, freq) * 2 + pd.Timedelta(days=7)
        context = _trading_index(start - padding, start, freq)
        needed = int(ffill_max_gap // freq) + 1
        context_start = context[-needed] if len(context) >= needed else start - padding
        context_start = max(start_dt, context_start)
        rebuilt = _resample_from_ticks(
            ccy_pair, context_start, end, freq, ffill_max_gap,
            extend_to_end=any(frame.index.min() >= end for frame in results),
            **options,
        )
        if not rebuilt.empty:
            rebuilt = rebuilt.loc[(rebuilt.index >= start) & (rebuilt.index < end)]
        partials = [
            frame for frame in partial_caches
            if frame is not None and not frame.empty
        ]
        if rebuilt.empty:
            if partials:
                warnings.warn(
                    f"Incomplete {source or DEFAULT_SOURCE} cache for {ccy_pair} "
                    f"in {start}..{end}; no ticks available to rebuild it. "
                    "Returning the available cached rows only.",
                    RuntimeWarning, stacklevel=2,
                )
                results.extend(partials)
            elif results:
                warnings.warn(
                    f"No {source or DEFAULT_SOURCE} cache or ticks for {ccy_pair} "
                    f"in {start}..{end}; returning the other available periods.",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            # A sparse/shorter source must not erase earlier or later cached
            # history. Within the rebuilt grid, even NaNs are authoritative.
            retained = [frame.loc[~frame.index.isin(rebuilt.index)] for frame in partials]
            retained = [frame for frame in retained if not frame.empty]
            if retained:
                warnings.warn(
                    f"Ticks do not cover all cached rows for {ccy_pair} "
                    f"in {start}..{end}; retaining cached rows outside the rebuilt range.",
                    RuntimeWarning, stacklevel=2,
                )
                results.extend(retained)
            results.append(rebuilt)

    if not results:
        return pd.DataFrame(columns=["bid", "ask", "mid"])
    result = pd.concat(results).sort_index(kind="stable")
    result = result[~result.index.duplicated(keep="last")]
    # Fill only the *index* between available slices. Filling prices again
    # would extend the budget of values already forward-filled in a cache.
    grid = _trading_index(result.index.min(), result.index.max() + freq, freq)
    result = result.reindex(result.index.union(grid).rename(result.index.name))
    return result.loc[(result.index >= start_dt) & (result.index < end_dt)]


def _resample_from_ticks(
    ccy_pair, start_dt, end_dt, freq, ffill_max_gap, *, raw_data_dir=None,
    tick_data_dir=None, prepared_raw_dir=None, source=DEFAULT_SOURCE,
    source_timezone=None, timestamp_format=None, extend_to_end=False,
):
    """Read right-closed bars with the preceding bar's input-file context."""

    raw_time_window = None
    if not pd.Timedelta(hours=1) % freq:
        bounds = _trading_bounds(start_dt, end_dt, freq)
        if bounds is None:
            return pd.DataFrame(columns=["bid", "ask", "mid"])
        # A following cache explicitly establishes this span's grid endpoint,
        # so out-of-session ticks beyond its last bar cannot contribute. For
        # standalone raw reads, keep the original window: even out-of-session
        # ticks can establish the extent of the subsequently filtered grid.
        # UTC file dates also avoid reopening the prior year's ZIP merely for
        # fill context that is already in the next UTC year.
        raw_time_window = (
            start_dt - freq,
            bounds[1] if extend_to_end else end_dt - pd.Timedelta(1, "ns"),
        )
    resampled_frames = []
    with quote_input_files(
        ccy_pair, (start_dt - freq).date(), end_dt.date(), tick_data_dir=tick_data_dir,
        raw_data_dir=raw_data_dir, prepared_raw_dir=prepared_raw_dir, source=source,
        raw_time_window=raw_time_window,
    ) as (file_paths, processed):
        for path in file_paths:
            frames = (
                _iter_processed_tick_file(path, ANNUAL_TICK_CHUNK_SIZE) if processed else
                _iter_tick_quote_file(path, source_timezone=source_timezone, source=source,
                                     timestamp_format=timestamp_format,
                                     chunksize=ANNUAL_TICK_CHUNK_SIZE)
            )
            with closing(frames):
                resampled_df = _resample_tick_frames(frames, start_dt, end_dt, freq)
            if resampled_df is not None:
                resampled_frames.append(resampled_df)

    if not resampled_frames:
        return pd.DataFrame(columns=["bid", "ask", "mid"])
    # A boundary bar can contain ticks from two files as well as two chunks.
    combined = pd.concat(resampled_frames).groupby(level=0, sort=True).last()
    combined["mid"] = (combined["bid"] + combined["ask"]) / 2
    if extend_to_end:
        # A following cached slice establishes the far side of this gap.
        # Extend before filling so the last raw quote gets only its remaining
        # fill budget, including at the end of a missing year.
        grid = _trading_index(start_dt, end_dt, freq)
        if not grid.empty:
            combined = combined.reindex(combined.index.union(grid[-1:]))
    return _finalize_resampled_frames([combined], freq, ffill_max_gap)


def _resample_tick_frames(frames, start_dt, end_dt, freq):
    """Merge partial bars with the same last-non-null rules as a full read."""
    file_frames = []
    for df in frames:
        # Pruning before resampling is phase-safe on hour-dividing grids.
        # Other frequencies retain pandas' original per-chunk day origin.
        if not pd.Timedelta(hours=1) % freq:
            df = df.loc[(df.index > start_dt - freq) & (df.index < end_dt)]
        if df.empty:
            continue
        resampled_df = df.resample(freq, label="right", closed="right").last()
        resampled_df = resampled_df.loc[
            (resampled_df.index >= start_dt) & (resampled_df.index < end_dt)
        ]
        if not resampled_df.empty:
            file_frames.append(resampled_df)
    if not file_frames:
        return None
    resampled_df = pd.concat(file_frames).groupby(level=0, sort=True).last()
    resampled_df["mid"] = (resampled_df["bid"] + resampled_df["ask"]) / 2
    return resampled_df


def get_resampled_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    freq: pd.Timedelta = pd.Timedelta(seconds=15),
    ffill_max_gap: pd.Timedelta = pd.Timedelta("10min"),
    max_workers: int | None = None,
    *,
    cache_dir: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    tick_data_dir: str | Path | None = None,
    prepared_raw_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
    source_timezone: str | None = None,
    timestamp_format: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load every pair from the selected source with identical loader options."""
    def load_one(ccy_pair: str) -> pd.DataFrame:
        return get_resampled_quote(
            ccy_pair,
            start_date,
            end_date,
            freq,
            ffill_max_gap,
            cache_dir=cache_dir,
            raw_data_dir=raw_data_dir,
            tick_data_dir=tick_data_dir,
            prepared_raw_dir=prepared_raw_dir,
            source=source,
            source_timezone=source_timezone,
            timestamp_format=timestamp_format,
            use_cache=use_cache,
        )

    if len(ccy_pairs) <= 1:
        return {ccy_pair: load_one(ccy_pair) for ccy_pair in ccy_pairs}

    if max_workers is None:
        max_workers = min(len(ccy_pairs), max(1, min(4, os.cpu_count() or 1)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda ccy_pair: (
                ccy_pair,
                load_one(ccy_pair),
            ),
            ccy_pairs,
        )
        return {ccy_pair: df for ccy_pair, df in results}


def _finalize_resampled_frames(
    frames: list[pd.DataFrame],
    freq: pd.Timedelta,
    ffill_max_gap: pd.Timedelta,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["bid", "ask", "mid"])

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    full = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, tz="US/Eastern"
    )
    df = df.reindex(full[_trading_mask(full)])

    limit = int(ffill_max_gap // pd.Timedelta(freq))
    if limit > 0:
        df = df.ffill(limit=limit)

    return df


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
    *,
    ffill_max_gap: pd.Timedelta = pd.Timedelta("10min"),
    raw_data_dir: str | Path | None = None,
    tick_data_dir: str | Path | None = None,
    prepared_raw_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
    source_timezone: str | None = None,
    timestamp_format: str | None = None,
    use_cache: bool = False,
    overwrite: bool = False,
) -> dict[str, dict[int, Path]]:
    """Write source-prefixed Eastern caches, resampling ticks by default.

    HISTDATA is the default output prefix even when the raw CSVs retain their
    original DAT_ASCII names. Resampling, forward filling and prices are
    independent of the filename prefix. DUKASCOPY Unix-millisecond inputs use
    the same headerless YYYYMMDD HHmmssSSS,bid,ask cache format and Eastern
    timezone as before. Read normalized ticks when available, otherwise raw
    ZIPs; explicitly setting raw_data_dir forces regeneration from raw ZIPs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = DEFAULT_SOURCE if source is None else source
    freq_code = _resampled_freq_code(freq)
    source_settings = get_raw_source_settings(source)

    saved_paths = {}

    for ccy_pair in ccy_pairs:
        df = get_resampled_quote(
            ccy_pair=ccy_pair,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            ffill_max_gap=ffill_max_gap,
            cache_dir=output_dir,
            raw_data_dir=raw_data_dir,
            tick_data_dir=tick_data_dir,
            prepared_raw_dir=prepared_raw_dir,
            source=source,
            source_timezone=source_timezone,
            timestamp_format=timestamp_format,
            use_cache=use_cache,
        )
        year_frames = (
            []
            if df.empty
            else [
                (int(year), year_df)
                for year, year_df in df.groupby(df.index.year)
            ]
        )
        output_pair = source_settings.format_cache_pair(ccy_pair)
        year_paths = {
            year: output_dir / f"{source}_{output_pair}_{freq_code}_{year}.csv"
            for year, _ in year_frames
        }

        existing_paths = [path for path in year_paths.values() if path.exists()]
        if existing_paths and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {existing_paths[0]}"
            )

        for year, year_df in year_frames:
            year_path = year_paths[year]
            output_df = year_df[["bid", "ask"]].copy()
            output_df.index = _format_histdata_timestamp_index(year_df.index)
            temporary_path = year_path.with_suffix(year_path.suffix + ".tmp")
            try:
                output_df.to_csv(temporary_path, header=False)
                temporary_path.replace(year_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        saved_paths[ccy_pair] = year_paths

    return saved_paths
