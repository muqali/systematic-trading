"""Shared storage rules for processed quotes, never vendor/raw source files."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_CACHE_FORMAT = "parquet"
CACHE_FORMATS = ("parquet", "csv")
CACHE_TIMEZONE = "US/Eastern"


def validate_cache_format(file_format: str) -> str:
    if file_format not in CACHE_FORMATS:
        raise ValueError(f"file_format must be one of {CACHE_FORMATS}, got {file_format!r}")
    return file_format


def resolve_quote_cache(path: Path) -> Path | None:
    """Select one representation, preferring Parquet unless CSV is newer.

    This permits incremental migration without double-reading rows or hiding
    a subsequently refreshed legacy CSV. Equal modification times favour
    Parquet. Corrupt selected files are handled by the calling loader, not
    silently replaced with an older representation here.
    """
    candidates = [path.with_suffix(f".{suffix}") for suffix in CACHE_FORMATS]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    return max(existing, key=lambda p: (p.stat().st_mtime_ns, p.suffix == ".parquet"), default=None)


def _quote_frame(frame: pd.DataFrame, *, reading: bool = False) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Parquet quotes require a timezone-aware DatetimeIndex.")
    if frame.index.hasnans:
        raise ValueError("Parquet quotes must not contain NaT timestamps.")
    if not {"bid", "ask"}.issubset(frame.columns):
        raise ValueError("Parquet quotes require bid and ask columns.")
    if reading and any(frame[column].dtype != "float64" for column in ("bid", "ask")):
        raise ValueError("Parquet bid and ask must be stored as float64.")
    result = frame[["bid", "ask"]].astype("float64")
    result.index = frame.index.tz_convert(CACHE_TIMEZONE).rename("timestamp")
    return result


def read_parquet_quotes(path: Path) -> pd.DataFrame:
    # pandas includes the stored index even with a price-column projection.
    return _quote_frame(pd.read_parquet(path, engine="pyarrow", columns=["bid", "ask"]), reading=True)


def iter_parquet_quotes(path: Path, chunksize: int):
    """Bound memory independently of a tick file's total size/row groups."""
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(batch_size=chunksize, columns=["timestamp", "bid", "ask"]):
            yield _quote_frame(batch.to_pandas(), reading=True)


def write_parquet_quotes(frame: pd.DataFrame, path: Path):
    _quote_frame(frame).to_parquet(path, engine="pyarrow", compression="snappy", index=True)


class ParquetQuoteWriter:
    """Append tick chunks as row groups; publish only after the writer closes."""

    def __init__(self, path: Path):
        self.path = path
        self.writer = None

    def __enter__(self):
        return self

    def write(self, frame: pd.DataFrame):
        table = pa.Table.from_pandas(_quote_frame(frame), preserve_index=True)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)

    def __exit__(self, *exc):
        if self.writer is not None:
            self.writer.close()


@contextmanager
def atomic_quote_output(target: Path, *, overwrite: bool):
    """Stage on the destination filesystem and publish atomically per file."""
    with tempfile.TemporaryDirectory(prefix=".quote-export-", dir=target.parent) as staging:
        temporary = Path(staging) / target.name
        yield temporary
        if overwrite:
            temporary.replace(target)
        else:
            # Unlike replace(), link() cannot clobber a concurrent writer.
            os.link(temporary, target)
