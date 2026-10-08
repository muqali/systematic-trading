"""Read-only source-priority composition of normalized resampled quotes.

Defaults to HISTDATA with DUKASCOPY fallback, except USDCNH and USDTHB,
which use DUKASCOPY directly for all years.

This layer selects complete bid/ask rows; it never resamples, forward-fills,
or saves prices itself. Source-specific storage remains the responsibility
of resampled_data_loader and tick_data_loader.
"""

import datetime as dt
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data.resampled_data_loader import (
    CACHE_TIMEZONE,
    _resampled_request_bounds,
    _trading_index,
    get_resampled_quote,
)
from data.tick_data_loader import get_raw_source_settings

DEFAULT_SOURCES = ("HISTDATA", "DUKASCOPY")
_DEFAULT_INSTRUMENT_SOURCES = {
    "USDCNH": ("DUKASCOPY",),
    "USDTHB": ("DUKASCOPY",),
}
_SOURCE_OPTION_NAMES = frozenset({
    "cache_dir", "tick_data_dir", "raw_data_dir", "prepared_raw_dir",
    "source_timezone", "timestamp_format",
})


def _source_priority(pair, sources, source_priority_by_instrument):
    routes = (
        _DEFAULT_INSTRUMENT_SOURCES
        if source_priority_by_instrument is None
        else {key.upper(): value for key, value in source_priority_by_instrument.items()}
    )
    selected = routes.get(pair.upper(), sources)
    if isinstance(selected, str) or not selected:
        raise ValueError("Source priority must be a nonempty sequence of source names.")
    selected = tuple(selected)
    if len(set(selected)) != len(selected):
        raise ValueError("Source priority must not contain duplicates.")
    for source in selected:
        if not isinstance(source, str):
            raise ValueError("Source names must be explicit registered strings.")
        get_raw_source_settings(source)
    return selected


def _validate_source_options(source_options):
    for source, options in source_options.items():
        if not isinstance(source, str):
            raise ValueError("Source names must be explicit registered strings.")
        get_raw_source_settings(source)
        unsupported = set(options) - _SOURCE_OPTION_NAMES
        if unsupported:
            raise ValueError(f"Unsupported options for {source}: {sorted(unsupported)}")


def _complete_quotes(frame):
    """Finite bid and ask are required; crossed quotes remain valid quotes."""
    return np.isfinite(frame[["bid", "ask"]].to_numpy(dtype=float)).all(axis=1)


def _empty_quotes(include_source):
    frame = pd.DataFrame(
        index=pd.DatetimeIndex([], tz=CACHE_TIMEZONE, name="timestamp"),
        columns=["bid", "ask", "mid"], dtype=float,
    )
    if include_source:
        frame["source"] = pd.Series(index=frame.index, dtype="string")
    return frame


def get_composite_resampled_quote(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    freq: pd.Timedelta = pd.Timedelta("1min"),
    ffill_max_gap: pd.Timedelta = pd.Timedelta("10min"),
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
    source_priority_by_instrument: Mapping[str, Sequence[str]] | None = None,
    include_source: bool = False,
    cache_dir: str | Path | None = None,
    tick_data_dir: str | Path | None = None,
    use_cache: bool = True,
    source_options: Mapping[str, Mapping] | None = None,
) -> pd.DataFrame:
    """Prefer HISTDATA, filling missing quotes from DUKASCOPY by default.

    USDCNH and USDTHB default directly to DUKASCOPY for all years, without
    probing HISTDATA. Instrument matching is case-insensitive. Passing
    source_priority_by_instrument={} disables this default routing; another
    mapping replaces it. Sources are ordered from highest to lowest priority.
    Each source is loaded at most once, and lower priorities are skipped when
    the requested trading grid already has complete quotes.
    The base loader checks file availability in the provider's calendar before
    extracting raw ZIPs. Sparse backups are matched on their available rows,
    without expanding each provider to another full requested grid.

    A missing timestamp or either non-finite price triggers whole-row fallback:
    bid and ask always come from the same source and mid is recomputed. Crossed
    quotes and existing forward-filled prices are preserved. No extra filling,
    price adjustment, backfilling, interpolation, or cache writes take place.
    If no source supplies both prices, all three prices remain NaN. Once any
    source has rows, the result uses the full requested Eastern trading grid;
    if every source is empty, an empty frame is returned, like the base loader.

    include_source adds a string column identifying each selected provider
    (NA for unresolved rows). It does not distinguish observed from previously
    forward-filled prices, which existing caches do not record.

    Supported frequencies divide one hour exactly (e.g. S15, M1, M15, H1).
    This ensures identical bar alignment across providers and DST boundaries.
    Date-window and weekend rules match get_resampled_quote. Per-source raw
    directories/clock overrides belong in source_options, never shared across
    providers. ffill_max_gap/use_cache are passed to the single-source loader;
    changing the fill limit cannot undo filling already stored in a cache.
    """
    freq, ffill_max_gap = pd.Timedelta(freq), pd.Timedelta(ffill_max_gap)
    if pd.isna(freq) or freq <= pd.Timedelta(0):
        raise ValueError("freq must be positive.")
    if pd.Timedelta(hours=1) % freq:
        raise ValueError("Composite frequency must divide one hour exactly.")
    if pd.isna(ffill_max_gap) or ffill_max_gap < pd.Timedelta(0):
        raise ValueError("ffill_max_gap must be nonnegative.")
    start, end = _resampled_request_bounds(start_date, end_date)
    priority = _source_priority(ccy_pair, sources, source_priority_by_instrument)
    source_options = {} if source_options is None else source_options
    _validate_source_options(source_options)
    index = _trading_index(start, end, freq).rename("timestamp")
    if index.empty:
        return _empty_quotes(include_source)

    prices = np.full((len(index), 2), np.nan)
    chosen_source = np.full(len(index), None, dtype=object) if include_source else None
    filled = np.zeros(len(index), dtype=bool)
    any_rows = False
    for source in priority:
        options = {"cache_dir": cache_dir, "tick_data_dir": tick_data_dir}
        options.update(source_options.get(source, {}))
        candidate = get_resampled_quote(
            ccy_pair, start_date, end_date, freq, ffill_max_gap,
            source=source, use_cache=use_cache, **options,
        )
        if candidate.empty:
            continue
        if (
            not isinstance(candidate.index, pd.DatetimeIndex)
            or candidate.index.tz is None or not candidate.index.is_unique
        ):
            raise ValueError(f"{source} returned a non-unique or non-timezone-aware index.")
        positions = index.get_indexer(candidate.index)
        in_grid = positions >= 0
        any_rows |= in_grid.any()
        take = in_grid & _complete_quotes(candidate)
        rows = np.flatnonzero(take)
        rows = rows[~filled[positions[rows]]]
        target = positions[rows]
        prices[target] = candidate[["bid", "ask"]].to_numpy(dtype=float)[rows]
        if include_source:
            chosen_source[target] = source
        filled[target] = True
        if filled.all():
            break

    if not any_rows:
        return _empty_quotes(include_source)
    result = pd.DataFrame(prices, index=index, columns=["bid", "ask"])
    result["mid"] = (result.bid + result.ask) / 2
    if include_source:
        result["source"] = pd.Series(chosen_source, index=index, dtype="string")
    return result


def get_composite_resampled_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    freq: pd.Timedelta = pd.Timedelta("1min"),
    ffill_max_gap: pd.Timedelta = pd.Timedelta("10min"),
    max_workers: int | None = None,
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
    source_priority_by_instrument: Mapping[str, Sequence[str]] | None = None,
    include_source: bool = False,
    cache_dir: str | Path | None = None,
    tick_data_dir: str | Path | None = None,
    use_cache: bool = True,
    source_options: Mapping[str, Mapping] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load instruments concurrently, retaining the caller's dictionary keys.

    Uses get_composite_resampled_quote's routing: USDCNH and USDTHB default
    directly to DUKASCOPY for all years; other instruments prefer HISTDATA
    with DUKASCOPY fallback. source_priority_by_instrument overrides these
    defaults; an empty mapping disables both instrument-specific routes.
    """
    def load_one(pair):
        return get_composite_resampled_quote(
            pair, start_date, end_date, freq, ffill_max_gap,
            sources=sources, source_priority_by_instrument=source_priority_by_instrument,
            include_source=include_source, cache_dir=cache_dir,
            tick_data_dir=tick_data_dir, use_cache=use_cache, source_options=source_options,
        )

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive.")
    if len(ccy_pairs) <= 1:
        return {pair: load_one(pair) for pair in ccy_pairs}
    if max_workers is None:
        max_workers = min(len(ccy_pairs), max(1, min(4, os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return dict(zip(ccy_pairs, executor.map(load_one, ccy_pairs)))
