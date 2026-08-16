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

FWD_TICKERS_DICT = {
    "EURUSD": "EUR",
    "USDJPY": "JPY",
    "GBPUSD": "GBP",
    "USDCHF": "CHF",
    "USDNOK": "NOK",
    "USDSEK": "SEK",
    "AUDUSD": "AUD",
    "NZDUSD": "NZD",
    "USDCAD": "CAD",
    "USDCNH": "CNH",
    "USDSGD": "SGD",
    "USDTHB": "THB",
    "USDHKD": "HKD",
    "USDTRY": "TRY",
    "USDZAR": "ZAR",
    "USDILS": "ILS",
    "USDPLN": "PLN",
    "USDHUF": "HUF",
    "USDCZK": "CZK",
    "USDRON": "RON",
    "USDMXN": "MXN",
    "EURNOK": "EURNOK",
    "EURSEK": "EURSEK",
    "EURPLN": "EURPLN",
    "EURHUF": "EURHUF",
    "EURCZK": "EURCZK",
    "EURRON": "EURRON",
    "EURGBP": "EURGBP",
    "EURJPY": "EURJPY",
    "EURCHF": "EURCHF",
}

ASSET_TICKERS_DICT = {
    "bond": BOND_TICKERS_DICT,
    "equity": EQUITY_TICKERS_DICT,
}


def _gt_yield_ticker(ccy: str, tenor: str) -> str:
    if ccy == "USD":
        return f"GT{tenor}".removesuffix("Y")
    return f"GT{ccy}{tenor}"


def get_gt_yield(
    ccys: list[str],
    tenor: str,
    start_date: dt.date,
    end_date: dt.date,
) -> dict[str, pd.DataFrame]:
    if not ccys:
        return {}

    ccys = [ccy.strip().upper() for ccy in ccys]
    if any(not ccy for ccy in ccys):
        raise ValueError("ccys must be non-empty.")

    tenor = tenor.strip().upper()
    if not tenor:
        raise ValueError("tenor must be non-empty.")

    file_path = CSV_DATA_DIR / f"gt{tenor.lower()}.csv"
    requested_tickers = {ccy: _gt_yield_ticker(ccy, tenor) for ccy in ccys}

    available_columns = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig").columns
    missing_tickers = [
        ticker
        for ticker in requested_tickers.values()
        if ticker not in available_columns
    ]
    if missing_tickers:
        supported_tickers = ", ".join(
            sorted(col for col in available_columns if col != "date")
        )
        raise ValueError(
            f"Missing government yield columns in '{file_path.name}': {missing_tickers}. "
            f"Supported columns: {supported_tickers}."
        )

    df = pd.read_csv(
        file_path,
        usecols=["date", *requested_tickers.values()],
        encoding="utf-8-sig",
        na_values=["#N/A N/A"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["timestamp"] = (df["date"] + pd.Timedelta(hours=17)).dt.tz_localize("US/Eastern")
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns="date")

    start_ts = pd.Timestamp(start_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    end_ts = pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return {
        ccy: df[[ticker]].rename(columns={ticker: "yield"})
        for ccy, ticker in requested_tickers.items()
    }


def get_fwdpts(
    ccypairs: list[str],
    tenor: str,
    start_date: dt.date,
    end_date: dt.date,
) -> dict[str, pd.DataFrame]:
    if not ccypairs:
        return {}

    tenor = tenor.strip().upper()
    if not tenor:
        raise ValueError("tenor must be non-empty.")

    file_path = CSV_DATA_DIR / f"fwdpts{tenor.lower()}.csv"
    missing_ccypairs = [
        ccypair for ccypair in ccypairs if ccypair not in FWD_TICKERS_DICT
    ]
    if missing_ccypairs:
        supported_ccypairs = ", ".join(sorted(FWD_TICKERS_DICT))
        raise ValueError(
            f"Unsupported forward point currency pairs: {missing_ccypairs}. "
            f"Supported currency pairs: {supported_ccypairs}."
        )

    requested_tickers = {
        ccypair: f"{FWD_TICKERS_DICT[ccypair]}{tenor}" for ccypair in ccypairs
    }

    available_columns = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig").columns
    missing_tickers = [
        ticker
        for ticker in requested_tickers.values()
        if ticker not in available_columns
    ]
    if missing_tickers:
        supported_tickers = ", ".join(
            sorted(col for col in available_columns if col != "date")
        )
        raise ValueError(
            f"Missing forward point columns in '{file_path.name}': {missing_tickers}. "
            f"Supported columns: {supported_tickers}."
        )

    df = pd.read_csv(
        file_path,
        usecols=["date", *requested_tickers.values()],
        encoding="utf-8-sig",
        na_values=["#N/A N/A"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["timestamp"] = (df["date"] + pd.Timedelta(hours=17)).dt.tz_localize("US/Eastern")
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns="date")

    start_ts = pd.Timestamp(start_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    end_ts = pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return {
        ccypair: df[[ticker]].rename(columns={ticker: "close"})
        for ccypair, ticker in requested_tickers.items()
    }


def get_wmco_prices(
    ccy_pairs: list[str], start_date: dt.date, end_date: dt.date
) -> dict[str, pd.DataFrame]:
    if not ccy_pairs:
        return {}

    file_path = CSV_DATA_DIR / "wmco.csv"
    available_columns = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig").columns
    missing_pairs = [
        ccy_pair for ccy_pair in ccy_pairs if ccy_pair not in available_columns
    ]
    if missing_pairs:
        supported_pairs = ", ".join(
            sorted(col for col in available_columns if col != "date")
        )
        raise ValueError(
            f"Unsupported currency pairs: {missing_pairs}. Supported pairs: {supported_pairs}."
        )

    df = pd.read_csv(
        file_path,
        usecols=["date", *ccy_pairs],
        encoding="utf-8-sig",
    )
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["timestamp"] = (
        (df["date"] + pd.Timedelta(hours=16))
        .dt.tz_localize("Europe/London")
        .dt.tz_convert("US/Eastern")
    )
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns="date")

    start_ts = (
        (pd.Timestamp(start_date) + pd.Timedelta(hours=16))
        .tz_localize("Europe/London")
        .tz_convert("US/Eastern")
    )
    end_ts = (
        (pd.Timestamp(end_date) + pd.Timedelta(hours=16))
        .tz_localize("Europe/London")
        .tz_convert("US/Eastern")
    )
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return {
        ccy_pair: pd.DataFrame(
            {
                "bid": df[ccy_pair],
                "ask": df[ccy_pair],
                "mid": df[ccy_pair],
            },
            index=df.index,
        )
        for ccy_pair in ccy_pairs
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
    df["timestamp"] = (df["date"] + pd.Timedelta(hours=17)).dt.tz_localize("US/Eastern")
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns="date")

    start_ts = pd.Timestamp(start_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    end_ts = pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(hours=17)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return {
        ccy: df[[tickers_dict[ccy]]].rename(columns={tickers_dict[ccy]: "close"})
        for ccy in ccys
    }
