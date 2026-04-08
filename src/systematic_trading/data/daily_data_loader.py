from pathlib import Path
import pandas as pd
import datetime as dt


CSV_DATA_DIR = Path.home() / "Programming" / "data" / "daily"

BOND_TICKERS_DICT = {
    "USD": "SBUSL",
    "EUR": "SBEGEU",
    "JPY": "SBJYL",
    "GBP": "SBUKL",
    "CHF": "SBSZL",
    "NOK": "SBNKL",
    "SEK": "SBSKL",
    "AUD": "SBADL",
    "NZD": "SBNZL",
    "CAD": "SBCDL",
}

EQUITY_TICKERS_DICT = {
    "USD": "MSDUUS",
    "EUR": "MSDLEMU",
    "JPY": "MSDLJN",
    "GBP": "MSDLUK",
    "CHF": "MSDLSZ",
    "NOK": "MSDLNO",
    "SEK": "MSDLSW",
    "AUD": "MSDLAS",
    "NZD": "MSDLNZ",
    "CAD": "MSDLCA",
}

ASSET_TICKERS_DICT = {
    "bond": BOND_TICKERS_DICT,
    "equity": EQUITY_TICKERS_DICT,
}


def get_asset_index_prices(
    asset: str,
    ccys: list[str],
    start_date: dt.date,
    end_date: dt.date,
) -> dict[str, pd.DataFrame]:
    if not ccys:
        return {}

    asset = asset.lower()
    tickers_dict = ASSET_TICKERS_DICT.get(asset)
    if tickers_dict is None:
        supported_assets = ", ".join(sorted(ASSET_TICKERS_DICT))
        raise ValueError(
            f"Unsupported asset '{asset}'. Supported assets: {supported_assets}."
        )

    missing_ccys = [ccy for ccy in ccys if ccy not in tickers_dict]
    if missing_ccys:
        supported_ccys = ", ".join(sorted(tickers_dict))
        raise ValueError(
            f"Unsupported currencies for asset '{asset}': {missing_ccys}. "
            f"Supported currencies: {supported_ccys}."
        )

    file_path = CSV_DATA_DIR / f"{asset}.csv"
    requested_tickers = [tickers_dict[ccy] for ccy in ccys]

    df = pd.read_csv(
        file_path,
        usecols=["date", *requested_tickers],
        encoding="utf-8-sig",
    )
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["timestamp"] = (
        df["date"] + pd.Timedelta(hours=17)
    ).dt.tz_localize("US/Eastern")
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns="date")

    start_ts = pd.Timestamp(start_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    end_ts = pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return {
        ccy: df[[tickers_dict[ccy]]].rename(columns={tickers_dict[ccy]: "close"})
        for ccy in ccys
    }
