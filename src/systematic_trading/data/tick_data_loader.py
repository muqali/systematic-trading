import datetime as dt
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pandas as pd

from data.tick_archives import extract_csv_from_zip

CSV_DATA_DIR = Path.home() / "Programming" / "data" / "tickquote"
RAW_DATA_ROOT = Path.home() / "Documents" / "data"
TIMESTAMP_FORMAT = "%Y%m%d %H%M%S%f"
LONDON_SYNCED_SOURCE = "London"
HISTDATA_SOURCE = "HISTDATA"
DEFAULT_SOURCE = HISTDATA_SOURCE
DUKASCOPY_SOURCE = "DUKASCOPY"
DUKASCOPY_DATA_DIR = RAW_DATA_ROOT / DUKASCOPY_SOURCE
UNIX_MILLISECONDS_FORMAT = "unix_ms"
DUKASCOPY_TIMESTAMP_FORMAT = UNIX_MILLISECONDS_FORMAT
ANNUAL_TICK_CHUNK_SIZE = 500_000


TARGET_TIMEZONE = "US/Eastern"
PROCESSED_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"


def dukascopy_tick_filename(ccy_pair: str, year: int) -> str:
    """Annual MarketData Hub name; next January 1 is the exclusive end date."""
    return f"{ccy_pair.lower()}_tick_{year}0101_{year + 1}0101.csv"


def _locate_monthly_files(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    data_dir: Path,
    source: str,
    *,
    prefixes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Select the first existing prefix for each month, without duplicates."""
    prefixes = (source,) if prefixes is None else prefixes
    paths = []
    start_month = start_date.replace(day=1)
    end_month = end_date.replace(day=1)
    current_month = start_month

    while current_month <= end_month:
        for prefix in prefixes:
            file_name = f"{prefix}_{ccy_pair}_T_{current_month.strftime('%Y%m')}.csv"
            file_path = data_dir / file_name
            if file_path.exists():
                paths.append(file_path)
                break

        current_month = (current_month.replace(day=28) + dt.timedelta(days=4)).replace(
            day=1
        )

    return paths


def _locate_dukascopy_files(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    data_dir: Path,
    source: str,
) -> list[Path]:
    """Accept only canonical annual files, not old CLI or partial downloads."""
    return [
        path
        for year in range(start_date.year, end_date.year + 1)
        if (path := data_dir / dukascopy_tick_filename(ccy_pair, year)).is_file()
    ]


FileLocator = Callable[[str, dt.date, dt.date, Path, str], list[Path]]


def _utc_file_dates(start: pd.Timestamp, end: pd.Timestamp) -> tuple[dt.date, dt.date]:
    """Inclusive file-calendar dates for an absolute tick-input window."""
    return start.tz_convert("UTC").date(), end.tz_convert("UTC").date()


def _locate_histdata_archives(
    pair: str, start: dt.date, end: dt.date, directory: Path, source: str,
) -> list[Path]:
    paths = []
    for month in pd.period_range(start, end, freq="M"):
        folder = directory / str(month.year) / pair.upper()
        filename = f"DAT_ASCII_{pair.upper()}_T_{month.strftime('%Y%m')}.zip"
        matches = sorted(folder.rglob(filename))
        if len(matches) > 1:
            raise ValueError(f"Duplicate monthly {source} archives: {matches}")
        paths.extend(matches)
    return paths


def _locate_dukascopy_archives(
    pair: str, start: dt.date, end: dt.date, directory: Path, source: str,
) -> list[Path]:
    return [
        path for year in range(start.year, end.year + 1)
        if (path := (directory / dukascopy_tick_filename(pair, year)).with_suffix(".zip")).is_file()
    ]


@dataclass(frozen=True)
class RawSourceSettings:
    """Provider rules consumed by the shared tick and resampling pipelines.

    Add a registry entry with ZIP and extracted-CSV locators for a new source.
    Columns are declared in timestamp/bid/ask semantic order: header
    names for headed CSVs, zero-based positions for headerless CSVs. Formats
    are pandas date formats or ``unix_ms``. Directory factories resolve late
    so callers can still configure the module's default directory constants.
    file_dates_for_window optionally maps an absolute tick window to the
    vendor's filename calendar (e.g. UTC years, not Eastern cache years).
    Without it, discovery retains the caller's inclusive calendar dates.
    """

    timezone: str | None = None
    timestamp_format: str | None = None
    default_data_dir: Callable[[], Path] | None = None
    find_files: FileLocator = _locate_monthly_files
    find_archives: FileLocator | None = None
    columns: tuple[str | int, str | int, str | int] = (0, 1, 2)
    has_header: bool = False
    require_ordered: bool = False
    format_cache_pair: Callable[[str], str] = str.upper
    file_dates_for_window: Callable[
        [pd.Timestamp, pd.Timestamp], tuple[dt.date, dt.date]
    ] | None = None


RAW_SOURCE_SETTINGS = {
    HISTDATA_SOURCE: RawSourceSettings(
        timezone=LONDON_SYNCED_SOURCE,
        timestamp_format=TIMESTAMP_FORMAT,
        find_archives=_locate_histdata_archives,
        find_files=partial(
            _locate_monthly_files, prefixes=("DAT_ASCII", HISTDATA_SOURCE)
        ),
    ),
    DUKASCOPY_SOURCE: RawSourceSettings(
        timezone="UTC",
        timestamp_format=DUKASCOPY_TIMESTAMP_FORMAT,
        default_data_dir=lambda: DUKASCOPY_DATA_DIR,
        find_files=_locate_dukascopy_files,
        find_archives=_locate_dukascopy_archives,
        columns=("timestamp", "bidPrice", "askPrice"),
        has_header=True,
        require_ordered=True,
        format_cache_pair=str.upper,
        file_dates_for_window=_utc_file_dates,
    ),
}


def get_raw_source_settings(source: str | None = DEFAULT_SOURCE) -> RawSourceSettings:
    """Require a registered provider; vendor filename prefixes are not sources."""
    source = DEFAULT_SOURCE if source is None else source
    try:
        return RAW_SOURCE_SETTINGS[source]
    except KeyError:
        raise ValueError(
            f"Unsupported source {source!r}; choose from {list(RAW_SOURCE_SETTINGS)}"
        ) from None


def _raw_source_directory(source: str, settings: RawSourceSettings) -> Path:
    return (
        RAW_DATA_ROOT / source
        if settings.default_data_dir is None else settings.default_data_dir()
    )


def locate_files(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    data_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
) -> list[Path]:
    """Find normalized monthly tick CSVs, never vendor files or raw archives."""
    source = DEFAULT_SOURCE if source is None else source
    settings = get_raw_source_settings(source)
    directory = CSV_DATA_DIR if data_dir is None else Path(data_dir)
    return _locate_monthly_files(
        settings.format_cache_pair(ccy_pair), start_date, end_date, directory, source
    )


@contextmanager
def quote_input_files(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    source: str | None = DEFAULT_SOURCE,
    tick_data_dir: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    prepared_raw_dir: str | Path | None = None,
    raw_time_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> Iterator[tuple[Iterable[Path], bool]]:
    """Yield (paths, processed): complete tick cache, otherwise raw ZIPs.

    raw_data_dir is a provider's ZIP root. prepared_raw_dir is only for the
    exporter's already-extracted temporary CSVs. Neither is a processed cache.
    Supplying either raw directory explicitly bypasses the processed cache.
    raw_time_window narrows vendor file discovery using its registered file
    calendar; its endpoints conservatively include boundary ticks. Processed
    cache selection continues to use the Eastern start_date/end_date.
    """
    source = DEFAULT_SOURCE if source is None else source
    settings = get_raw_source_settings(source)
    pair = settings.format_cache_pair(ccy_pair)
    if raw_data_dir is not None and prepared_raw_dir is not None:
        raise ValueError("Specify raw_data_dir or prepared_raw_dir, not both.")
    raw_start, raw_end = start_date, end_date
    if raw_time_window is not None and settings.file_dates_for_window is not None:
        raw_start, raw_end = settings.file_dates_for_window(*raw_time_window)
    if prepared_raw_dir is not None:
        paths = settings.find_files(
            pair, raw_start, raw_end, Path(prepared_raw_dir), source
        )
        yield paths, False
        return
    if raw_data_dir is None:
        paths = locate_files(pair, start_date, end_date, tick_data_dir, source)
        if len(paths) == len(pd.period_range(start_date, end_date, freq="M")):
            yield paths, True
            return
    directory = (
        _raw_source_directory(source, settings)
        if raw_data_dir is None else Path(raw_data_dir)
    )
    if settings.find_archives is None:
        raise ValueError(f"Raw source {source!r} must register a ZIP file locator.")
    archives = settings.find_archives(pair, raw_start, raw_end, directory, source)
    with tempfile.TemporaryDirectory(prefix=f"raw-ticks-{source}-") as staging:
        def extracted_paths():
            for archive in archives:
                path = extract_csv_from_zip(
                    archive, Path(staging), archive.with_suffix(".csv").name
                )
                yield path
                path.unlink()
        yield extracted_paths(), False


def _iter_processed_tick_file(
    path: Path, chunksize: int = ANNUAL_TICK_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Read the common headed schema; explicit offsets disambiguate DST folds."""
    with pd.read_csv(
        path, usecols=["timestamp", "bid", "ask"],
        dtype={"timestamp": "string", "bid": "float64", "ask": "float64"},
        float_precision="round_trip", chunksize=chunksize,
    ) as reader:
        for df in reader:
            index = pd.to_datetime(
                df.pop("timestamp"), format=PROCESSED_TIMESTAMP_FORMAT, utc=True
            )
            df.index = pd.DatetimeIndex(index).as_unit("us").tz_convert(TARGET_TIMEZONE)
            df["mid"] = (df["bid"] + df["ask"]) / 2
            yield df


def _format_processed_tick_index(index: pd.DatetimeIndex) -> pd.Index:
    """Format local wall time and offsets separately to avoid per-tick tz work."""
    eastern = index.tz_convert(TARGET_TIMEZONE)
    local = eastern.tz_localize(None)
    offsets = pd.Index((local - eastern.tz_convert(None)).total_seconds().astype("int64"))
    labels = {}
    for seconds in offsets.unique():
        hours, remainder = divmod(abs(int(seconds)), 3600)
        minutes, seconds_part = divmod(remainder, 60)
        label = f"{'-' if seconds < 0 else '+'}{hours:02d}{minutes:02d}"
        labels[seconds] = label + (f"{seconds_part:02d}" if seconds_part else "")
    return pd.Index(
        local.strftime("%Y-%m-%d %H:%M:%S.%f") + offsets.map(labels), name="timestamp"
    )


def _read_quote_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[0, 1, 2],
        names=["timestamp", "bid", "ask"],
        header=None,
        dtype={"timestamp": "string", "bid": "float64", "ask": "float64"},
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    return df


def _localize_raw_timestamp_index(
    index: pd.DatetimeIndex,
    source_timezone: str,
) -> pd.DatetimeIndex:
    if index.tz is not None:
        raise ValueError("Raw timestamp index must be timezone-naive.")

    if source_timezone == LONDON_SYNCED_SOURCE:
        # Empirical HistData convention: raw wall time plus five hours tracks
        # Europe/London local time, including the UK daylight-saving calendar.
        return (
            (index + pd.Timedelta(hours=5))
            .tz_localize(
                "Europe/London",
                ambiguous="NaT",
                nonexistent="NaT",
            )
            .tz_convert(TARGET_TIMEZONE)
        )

    return index.tz_localize(source_timezone).tz_convert(TARGET_TIMEZONE)


def _read_tick_quote_file(
    path: Path,
    source_timezone: str | None = None,
    *,
    source: str | None = DEFAULT_SOURCE,
    timestamp_format: str | None = None,
) -> pd.DataFrame:
    """Read a raw file using its provider's schema, clock and timestamp format."""
    return next(
        _iter_tick_quote_file(
            path, source_timezone, source=source, timestamp_format=timestamp_format
        )
    )


def _resolve_read_settings(
    source: str,
    source_timezone: str | None,
    timestamp_format: str | None,
) -> RawSourceSettings:
    """Validate raw-only overrides once per file, before opening the CSV."""
    settings = get_raw_source_settings(source)
    if (
        settings.timestamp_format is not None
        and timestamp_format is not None
        and timestamp_format != settings.timestamp_format
    ):
        raise ValueError(
            f"{source} requires timestamp_format={settings.timestamp_format!r}; "
            f"received {timestamp_format!r}."
        )
    timezone = source_timezone if source_timezone is not None else settings.timezone
    date_format = (
        timestamp_format if timestamp_format is not None else settings.timestamp_format
    )
    if timezone is None or date_format is None:
        raise ValueError(
            f"Unknown raw source {source!r}: specify source_timezone and timestamp_format."
        )
    if date_format == UNIX_MILLISECONDS_FORMAT and timezone != "UTC":
        raise ValueError(
            f"{source} Unix millisecond timestamps require source_timezone='UTC'; "
            f"received {timezone!r}."
        )
    return replace(settings, timezone=timezone, timestamp_format=date_format)


def _csv_read_options(settings: RawSourceSettings, chunksize: int | None) -> dict:
    """Read only the timestamp and prices; optional volume fields stay on disk."""
    timestamp, bid, ask = settings.columns
    return {
        "usecols": list(settings.columns),
        "header": 0 if settings.has_header else None,
        "dtype": {timestamp: "string", bid: "float64", ask: "float64"},
        "chunksize": chunksize,
    }


def _parse_raw_timestamp_index(
    timestamps: pd.Series,
    settings: RawSourceSettings,
    source: str,
    path: Path,
) -> pd.DatetimeIndex:
    """Convert the configured timestamp encoding to the common Eastern index."""
    if settings.timestamp_format == UNIX_MILLISECONDS_FORMAT:
        # Text validation avoids silently accepting wrong units or old formats.
        if not timestamps.str.fullmatch(r"[0-9]{13}", na=False).all():
            raise ValueError(
                f"Unparseable {source} timestamps in {path}; "
                f"expected format {settings.timestamp_format!r} "
                "(13-digit UTC Unix milliseconds)."
            )
        parsed = pd.to_datetime(timestamps.astype("int64"), unit="ms", utc=True)
        # Match the index resolution of compact-date parsing under pandas 3.
        return pd.DatetimeIndex(parsed).as_unit("us").tz_convert(TARGET_TIMEZONE)

    parsed = pd.to_datetime(
        timestamps, format=settings.timestamp_format, errors="coerce"
    )
    return _localize_raw_timestamp_index(pd.DatetimeIndex(parsed), settings.timezone)


def _check_timestamp_order(
    index: pd.DatetimeIndex,
    previous: pd.Timestamp | None,
    source: str,
    path: Path,
) -> pd.Timestamp | None:
    """Allow repeated timestamps, but reject time going backwards across chunks."""
    if index.empty:
        return previous
    if not index.is_monotonic_increasing or (
        previous is not None and index[0] < previous
    ):
        raise ValueError(f"{source} ticks must be timestamp-ordered: {path}")
    return index[-1]


def _iter_tick_quote_file(
    path: Path,
    source_timezone: str | None = None,
    *,
    source: str | None = DEFAULT_SOURCE,
    timestamp_format: str | None = None,
    chunksize: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Read raw quotes with one configured format per source (not cache CSVs).

    Explicit formats may repeat a registered source's format, but cannot
    override it. No alternate-format detection is performed.
    Unix millisecond timestamps are absolute instants and cannot be
    reinterpreted as another source clock.
    """
    source = DEFAULT_SOURCE if source is None else source
    settings = _resolve_read_settings(source, source_timezone, timestamp_format)
    column_names = dict(zip(settings.columns, ("timestamp", "bid", "ask")))
    reader = pd.read_csv(path, **_csv_read_options(settings, chunksize))
    frames = [reader] if chunksize is None else reader
    last_timestamp = None
    try:
        for df in frames:
            df = df.rename(columns=column_names)
            index = _parse_raw_timestamp_index(df["timestamp"], settings, source, path)
            df = df[["bid", "ask"]]
            # Preserve each source's bid/ask fields, including crossed quotes.
            df.index = index
            df = df[~df.index.isna()]
            if settings.require_ordered:
                last_timestamp = _check_timestamp_order(
                    df.index, last_timestamp, source, path
                )
            df["mid"] = (df["bid"] + df["ask"]) / 2
            yield df
    finally:
        if chunksize is not None:
            reader.close()


def _get_tick_quote_generator(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    source_timezone: str | None = None,
    data_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
    timestamp_format: str | None = None,
    *,
    raw_data_dir: str | Path | None = None,
    prepared_raw_dir: str | Path | None = None,
) -> Iterator[pd.DataFrame]:
    """Read normalized ticks, falling back to ZIPs without writing a cache."""
    with quote_input_files(
        ccy_pair, start_date, end_date, source=source, tick_data_dir=data_dir,
        raw_data_dir=raw_data_dir, prepared_raw_dir=prepared_raw_dir,
    ) as (paths, processed):
        for path in paths:
            frames = (
                _iter_processed_tick_file(path, ANNUAL_TICK_CHUNK_SIZE) if processed else
                _iter_tick_quote_file(path, source_timezone, source=source,
                                     timestamp_format=timestamp_format,
                                     chunksize=ANNUAL_TICK_CHUNK_SIZE)
            )
            with closing(frames):
                for df in frames:
                    if not df.empty:
                        yield df


def get_tick_quote(
    ccy_pair: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    source_timezone: str | None = None,
    data_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
    timestamp_format: str | None = None,
) -> pd.DataFrame:
    """Load Eastern ticks from processed files or the configured source ZIPs.

    data_dir contains processed monthly SOURCE_PAIR_T_YYYYMM.csv files.
    It defaults to Programming/data/tickquote and does not select raw inputs.
    Missing cache months fall back to the source's configured ZIP folder,
    normally Documents/data/<source>.
    Clock/format overrides apply only to raw inputs, never processed ticks.
    Reads do not populate persistent files; use save_tick_quotes to do that.
    Filter each chunk before retaining it, so memory scales with the requested
    ticks plus the read chunk, not whole source files. Files are still scanned
    completely, including timestamp validation outside the requested window.
    """
    start_dt = pd.Timestamp(
        start_date.year, start_date.month, start_date.day, 17, tz="US/Eastern"
    ) - pd.Timedelta(days=1)

    end_dt = pd.Timestamp(
        end_date.year, end_date.month, end_date.day, 17, tz="US/Eastern"
    )

    frames = []
    empty_result = None
    with closing(
        _get_tick_quote_generator(
            ccy_pair,
            start_date - dt.timedelta(days=1),
            end_date,
            source_timezone=source_timezone,
            data_dir=data_dir,
            source=source,
            timestamp_format=timestamp_format,
        )
    ) as chunks:
        for df in chunks:
            selected = df.loc[(df.index >= start_dt) & (df.index < end_dt)]
            if not selected.empty:
                frames.append(selected.copy())
            elif empty_result is None:
                # Preserve the typed Eastern index when files exist but contain
                # no matching ticks, without retaining any out-of-window data.
                empty_result = selected.copy()
    if not frames:
        return (
            empty_result if empty_result is not None
            else pd.DataFrame(columns=["bid", "ask", "mid"])
        )
    return pd.concat(frames).sort_index(kind="stable")


def get_tick_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    max_workers: int | None = None,
    *,
    source_timezone: str | None = None,
    data_dir: str | Path | None = None,
    source: str | None = DEFAULT_SOURCE,
    timestamp_format: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load each pair's Eastern ticks using the same options as get_tick_quote.

    data_dir selects processed tick files only. Missing cache months fall back
    to the configured source ZIP folder, normally Documents/data/<source>.
    """
    def load_one(ccy_pair: str) -> pd.DataFrame:
        return get_tick_quote(
            ccy_pair,
            start_date,
            end_date,
            source_timezone=source_timezone,
            data_dir=data_dir,
            source=source,
            timestamp_format=timestamp_format,
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


def save_tick_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path | None = None,
    *,
    source: str | None = DEFAULT_SOURCE,
    raw_data_dir: str | Path | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> dict[str, dict[str, Path]]:
    """Normalize raw ZIPs to monthly headed timestamp,bid,ask tick CSVs.

    Export complete Eastern calendar months intersecting the date range (not
    the 17:00 trading-day window used by get_tick_quote). Offset-bearing
    timestamps retain milliseconds and both instances of a DST repeated hour.
    Duplicate timestamps and crossed quotes are preserved; mid is derived on
    read. ZIPs remain untouched. Writes are staged, then atomically published
    per file only after the pair's input has been read successfully. resume
    skips existing monthly files; overwrite replaces them. These options are
    mutually exclusive. Missing source months are not manufactured or filled.
    """
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    source = DEFAULT_SOURCE if source is None else source
    settings = get_raw_source_settings(source)
    output_dir = CSV_DATA_DIR if output_dir is None else Path(output_dir)
    months = pd.period_range(start_date, end_date, freq="M")
    start = months[0].start_time.tz_localize(TARGET_TIMEZONE)
    end = (months[-1] + 1).start_time.tz_localize(TARGET_TIMEZONE)
    raw_root = (
        _raw_source_directory(source, settings)
        if raw_data_dir is None else Path(raw_data_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for pair in ccy_pairs:
        output_pair = settings.format_cache_pair(pair)
        targets = {
            month.strftime("%Y%m"):
            output_dir / f"{source}_{output_pair}_T_{month.strftime('%Y%m')}.csv"
            for month in months
        }
        for target in targets.values():
            if target.exists() and not (overwrite or resume):
                raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        skipped = {
            month: path for month, path in targets.items() if resume and path.exists()
        }
        if len(skipped) == len(targets):
            saved[pair] = skipped
            continue
        written = {}
        # Adjacent source dates can cross a month/year when converted to Eastern.
        with (
            tempfile.TemporaryDirectory(prefix=".tick-export-", dir=output_dir) as staging,
            closing(_get_tick_quote_generator(
                pair, (start - pd.Timedelta(days=1)).date(), end.date(),
                source=source, raw_data_dir=raw_root,
            )) as frames,
        ):
            for df in frames:
                df = df.loc[(df.index >= start) & (df.index < end), ["bid", "ask"]]
                if df.empty:
                    continue
                month_numbers = df.index.year * 100 + df.index.month
                for month_number, part in df.groupby(month_numbers, sort=False):
                    month = str(month_number)
                    if month in skipped:
                        continue
                    staged = Path(staging) / targets[month].name
                    part = part.copy()
                    part.index = _format_processed_tick_index(part.index)
                    part.to_csv(staged, mode="a", header=not staged.exists())
                    written[month] = targets[month]
            if not written and not skipped:
                raise FileNotFoundError(f"No raw {source} ticks for {pair} in {start_date}..{end_date}")
            for month, target in written.items():
                staged = Path(staging) / target.name
                if overwrite:
                    staged.replace(target)
                else:
                    # Atomic no-clobber publication, including concurrent writers.
                    os.link(staged, target)
        saved[pair] = skipped | written
    return saved
