import datetime as dt
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from data.tick_data_loader import get_tick_quote


CSV_DATA_DIR = Path.home() / "Programming" / "data" / "minutebar"


def locate_files(ccy_pair: str, start_date: dt.date, end_date: dt.date) -> list[Path]:
    paths = []
    start_year = start_date.replace(month=1, day=1)
    end_year = end_date.replace(month=1, day=1)
    current_year = start_year

    while current_year <= end_year:
        file_name = f"DAT_ASCII_{ccy_pair}_M1_{current_year.strftime('%Y')}.csv"
        file_path = CSV_DATA_DIR / file_name

        if file_path.exists():
            paths.append(file_path)

        current_year = (current_year + dt.timedelta(days=366)).replace(day=1)

    return paths


def _get_minute_bar_generator(ccy_pair: str, start_date: dt.date, end_date: dt.date):
    '''
    CSV Format from HistData.com:
    
    https://www.histdata.com/f-a-q/data-files-detailed-specification/

    Row Fields:
    DateTime Stamp;Bar OPEN Bid Quote;Bar HIGH Bid Quote;Bar LOW Bid Quote;Bar CLOSE Bid Quote;Volume

    DateTime Stamp Format:
    YYYYMMDD HHMMSS

    Legend:
    YYYY - Year
    MM - Month (01 to 12)
    DD - Day of the Month
    HH - Hour of the day (in 24h format)
    MM - Minute
    SS - Second, in this case it will be allways 00

    TimeZone: Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings adjustments
    '''

    file_paths = locate_files(ccy_pair, start_date, end_date)

    for path in file_paths:
        df = pd.read_csv(
            path,
            sep=";",
            usecols=[0, 1, 2, 3, 4],
            names=["timestamp", "open", "high", "low", "close"],
            header=None,
            index_col=0,
            parse_dates=True,
            date_format="%Y%m%d %H%M%S%f",
        )

        df = df.tz_localize("EST").tz_convert("US/Eastern")
        yield df


def get_minute_bar(
    ccy_pair: str, start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:

    df = pd.concat(
        _get_minute_bar_generator(ccy_pair, start_date - dt.timedelta(days=1), end_date)
    )

    start_dt = pd.Timestamp(
        start_date.year, start_date.month, start_date.day, 17, tz="US/Eastern"
    ) - pd.Timedelta(days=1)

    end_dt = pd.Timestamp(
        end_date.year, end_date.month, end_date.day, 17, tz="US/Eastern"
    )

    mask = (df.index >= start_dt) & (df.index < end_dt)

    return df.loc[mask]


def get_minute_bars(
    ccy_pairs: list[str], start_date: dt.date, end_date: dt.date
) -> dict[str, pd.DataFrame]:

    return {
        ccy_pair: get_minute_bar(ccy_pair, start_date, end_date)
        for ccy_pair in ccy_pairs
    }


def get_minute_quote(
    ccy_pair: str, start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:
    df = get_tick_quote(ccy_pair, start_date, end_date)

    # We only need minute-close quotes, so use last() instead of ohlc().
    df = df[["bid", "ask", "mid"]].resample("1min", label="right", closed="right").last()
    if df.empty:
        return df

    full = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min")

    mask = (
        (full.dayofweek < 4)
        | ((full.dayofweek == 4) & (full.hour < 17))
        | ((full.dayofweek == 6) & (full.hour >= 18))
    )

    full = full[mask]

    df = df.reindex(full)

    return df.ffill(limit=10)


def get_minute_quotes(
    ccy_pairs: list[str],
    start_date: dt.date,
    end_date: dt.date,
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    if len(ccy_pairs) <= 1:
        return {
            ccy_pair: get_minute_quote(ccy_pair, start_date, end_date)
            for ccy_pair in ccy_pairs
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda ccy_pair: (
                ccy_pair,
                get_minute_quote(ccy_pair, start_date, end_date),
            ),
            ccy_pairs,
        )
        return {ccy_pair: df for ccy_pair, df in results}
