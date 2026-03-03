import datetime as dt
import pandas as pd
from pathlib import Path

CSV_DATA_DIR = Path.home() / "Programming" / "data" / "tickquote"


def locate_files(ccy_pair: str, start_date: dt.date, end_date: dt.date) -> list[Path]:
    paths = []
    start_month = start_date.replace(day=1)
    end_month = end_date.replace(day=1)
    current_month = start_month

    while current_month <= end_month:
        file_name = f"DAT_ASCII_{ccy_pair}_T_{current_month.strftime('%Y%m')}.csv"
        file_path = CSV_DATA_DIR / file_name

        if file_path.exists():
            paths.append(file_path)

        current_month = (current_month.replace(day=28) + dt.timedelta(days=4)).replace(
            day=1
        )

    return paths


def _get_tick_quote_generator(ccy_pair: str, start_date: dt.date, end_date: dt.date):
    '''
    CSV Format from HistData.com:
    
    https://www.histdata.com/f-a-q/data-files-detailed-specification/

    Row Fields:
    DateTime Stamp,Bid Quote,Ask Quote,Volume

    DateTime Stamp Format:
    YYYYMMDD HHMMSSNNN

    Legend:
    YYYY - Year
    MM - Month (01 to 12)
    DD - Day of the Month
    HH - Hour of the day (in 24h format)
    MM - Minute
    SS - Second
    NNN - Millisecond
    '''

    file_paths = locate_files(ccy_pair, start_date, end_date)

    for path in file_paths:
        df = pd.read_csv(
            path,
            usecols=[0, 1, 2],
            names=["timestamp", "bid", "ask"],
            header=None,
            index_col=0,
            parse_dates=True,
            date_format="%Y%m%d %H%M%S%f",
        )

        df = df.tz_localize("EST").tz_convert("US/Eastern")
        df["mid"] = (df["bid"] + df["ask"]) / 2
        yield df


def get_tick_quote(
    ccy_pair: str, start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:

    df = pd.concat(
        _get_tick_quote_generator(ccy_pair, start_date - dt.timedelta(days=1), end_date)
    )

    start_dt = pd.Timestamp(
        start_date.year, start_date.month, start_date.day, 17, tz="US/Eastern"
    ) - pd.Timedelta(days=1)

    end_dt = pd.Timestamp(
        end_date.year, end_date.month, end_date.day, 17, tz="US/Eastern"
    )

    mask = (df.index >= start_dt) & (df.index < end_dt)

    return df.loc[mask]


def get_tick_quotes(
    ccy_pairs: list[str], start_date: dt.date, end_date: dt.date
) -> dict[str, pd.DataFrame]:

    return {
        ccy_pair: get_tick_quote(ccy_pair, start_date, end_date)
        for ccy_pair in ccy_pairs
    }
