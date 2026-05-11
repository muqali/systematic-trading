from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.resampled_data_loader import save_resampled_quotes


HISTDATA_ROOT = Path.home() / "Documents" / "data" / "histdata.com"
TICKQUOTE_DIR = Path.home() / "Programming" / "data" / "tickquote"


def _discover_ccy_pairs(year_dir: Path) -> list[str]:
    return sorted(path.name for path in year_dir.iterdir() if path.is_dir())


def _discover_zip_files(year_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in year_dir.rglob("DAT_ASCII_*_T_*.zip")
        if path.is_file()
    )


def _stage_zip_file(zip_path: Path, tickquote_dir: Path) -> Path:
    tickquote_dir.mkdir(parents=True, exist_ok=True)
    target_path = tickquote_dir / zip_path.name

    if zip_path.resolve() == target_path.resolve():
        return target_path

    if target_path.exists():
        return target_path

    shutil.copy2(zip_path, target_path)
    return target_path


def _extract_tick_csvs(zip_path: Path, tickquote_dir: Path) -> list[Path]:
    extracted_paths: list[Path] = []

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_name = Path(member.filename).name
            if not member_name.endswith(".csv"):
                continue
            if not member_name.startswith("DAT_ASCII_") or "_T_" not in member_name:
                continue

            target_path = tickquote_dir / member_name
            if target_path.exists():
                extracted_paths.append(target_path)
                continue

            with archive.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_paths.append(target_path)

    return extracted_paths


def _stage_and_extract_year(
    year_dir: Path,
    tickquote_dir: Path,
) -> tuple[list[str], list[Path], list[Path]]:
    ccy_pairs = _discover_ccy_pairs(year_dir)
    zip_paths = _discover_zip_files(year_dir)

    staged_zip_paths: list[Path] = []
    extracted_csv_paths: list[Path] = []
    for zip_path in zip_paths:
        staged_zip = _stage_zip_file(zip_path, tickquote_dir)
        staged_zip_paths.append(staged_zip)
        extracted_csv_paths.extend(_extract_tick_csvs(staged_zip, tickquote_dir))

    return ccy_pairs, staged_zip_paths, extracted_csv_paths


def _cleanup_staged_files(paths: list[Path]) -> int:
    deleted_count = 0
    for path in paths:
        if not path.exists():
            continue
        path.unlink()
        deleted_count += 1
    return deleted_count


def export_year(
    year: int,
    histdata_root: Path = HISTDATA_ROOT,
    tickquote_dir: Path = TICKQUOTE_DIR,
    freq: pd.Timedelta = pd.Timedelta(minutes=15),
    resampled_output_dir: str | Path | None = None,
) -> dict:
    year_dir = histdata_root / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory does not exist: {year_dir}")

    ccy_pairs, staged_zip_paths, extracted_csv_paths = _stage_and_extract_year(
        year_dir=year_dir,
        tickquote_dir=tickquote_dir,
    )
    zip_count = len(staged_zip_paths)
    extracted_count = len(extracted_csv_paths)

    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)

    try:
        if resampled_output_dir is None:
            saved_paths = save_resampled_quotes(ccy_pairs, start_date, end_date, freq)
        else:
            saved_paths = save_resampled_quotes(
                ccy_pairs,
                start_date,
                end_date,
                freq,
                output_dir=resampled_output_dir,
            )
    finally:
        deleted_staged_zip_count = _cleanup_staged_files(staged_zip_paths)
        deleted_extracted_csv_count = _cleanup_staged_files(extracted_csv_paths)

    return {
        "year": year,
        "ccy_pairs": ccy_pairs,
        "zip_count": zip_count,
        "extracted_csv_count": extracted_count,
        "deleted_staged_zip_count": deleted_staged_zip_count,
        "deleted_extracted_csv_count": deleted_extracted_csv_count,
        "saved_paths": saved_paths,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy HistData tick zip files into the tickquote directory, "
            "extract tick CSVs, generate resampled quote files, "
            "and then remove the staged zip copies and extracted CSVs."
        )
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="One or more calendar years to export, for example: --years 2024 2025",
    )
    parser.add_argument(
        "--histdata-root",
        type=Path,
        default=HISTDATA_ROOT,
        help=f"Root folder containing yearly histdata subfolders. Default: {HISTDATA_ROOT}",
    )
    parser.add_argument(
        "--tickquote-dir",
        type=Path,
        default=TICKQUOTE_DIR,
        help=f"Directory where monthly tick CSV files should be staged. Default: {TICKQUOTE_DIR}",
    )
    parser.add_argument(
        "--resampled-output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory used by save_resampled_quotes.",
    )
    parser.add_argument(
        "--freq",
        default="15min",
        help="Resample frequency understood by pandas.Timedelta. Default: 15min",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    freq = pd.Timedelta(args.freq)

    for year in args.years:
        result = export_year(
            year=year,
            histdata_root=args.histdata_root,
            tickquote_dir=args.tickquote_dir,
            freq=freq,
            resampled_output_dir=args.resampled_output_dir,
        )
        print(
            f"{year}: discovered {len(result['ccy_pairs'])} ccy pairs, "
            f"copied {result['zip_count']} tick zip files, "
            f"extracted {result['extracted_csv_count']} tick CSVs, "
            f"deleted {result['deleted_staged_zip_count']} staged zip files, "
            f"deleted {result['deleted_extracted_csv_count']} extracted tick CSVs."
        )


if __name__ == "__main__":
    main()
