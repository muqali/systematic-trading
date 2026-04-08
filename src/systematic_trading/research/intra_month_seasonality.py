from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _get_price_series(data: pd.DataFrame | pd.Series, price_col: str) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()

    if price_col not in data.columns:
        raise ValueError(f"Expected column {price_col!r} in input data.")

    return data[price_col].dropna()


def calculate_daily_close_returns(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    price_col: str = "mid",
    log_return: bool = True,
) -> pd.DataFrame:
    daily_returns = {}

    for ccy_pair, data in quote_dict.items():
        price_series = _get_price_series(data, price_col)
        if price_series.empty:
            continue

        daily_close = price_series.groupby(price_series.index.normalize()).last().dropna()
        if len(daily_close) < 2:
            continue

        if log_return:
            returns = np.log(daily_close).diff()
        else:
            returns = daily_close.pct_change()

        daily_returns[ccy_pair] = returns.dropna()

    if not daily_returns:
        return pd.DataFrame()

    return pd.DataFrame(daily_returns).sort_index()


def classify_intra_month_buckets(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.DatetimeIndex(index).normalize()
    frame = pd.DataFrame(index=dates.unique().sort_values())
    frame["month"] = frame.index.to_period("M")
    frame["month_day_count"] = frame.groupby("month").cumcount() + 1
    frame["month_size"] = frame.groupby("month")["month"].transform("size")
    frame["reverse_day_count"] = frame["month_size"] - frame["month_day_count"] + 1

    bucket = pd.Series("middle", index=frame.index, dtype="object")
    bucket[frame["month_day_count"] <= 7] = "first_7"
    bucket[(frame["month_day_count"] > 7) & (frame["reverse_day_count"] <= 7)] = "last_7"

    return bucket.reindex(dates)


def calculate_intra_month_bucket_returns(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    price_col: str = "mid",
    log_return: bool = True,
) -> dict[str, pd.DataFrame]:
    daily_returns = calculate_daily_close_returns(
        quote_dict=quote_dict,
        price_col=price_col,
        log_return=log_return,
    )
    if daily_returns.empty:
        return {}

    bucket_labels = classify_intra_month_buckets(daily_returns.index)
    bucketed_returns = {}

    for ccy_pair in daily_returns.columns:
        pair_returns = daily_returns[ccy_pair].dropna()
        if pair_returns.empty:
            continue

        pair_buckets = bucket_labels.reindex(pair_returns.index)
        pair_frame = pd.DataFrame({"return": pair_returns, "bucket": pair_buckets})

        bucketed = {}
        for bucket in ("first_7", "middle", "last_7"):
            bucketed[bucket] = pair_frame["return"].where(pair_frame["bucket"] == bucket)

        bucketed_returns[ccy_pair] = pd.DataFrame(bucketed, index=pair_returns.index)

    return bucketed_returns


def plot_intra_month_seasonality(
    bucketed_returns: dict[str, pd.DataFrame],
    instrument: str,
    figsize: tuple[float, float] = (10, 5),
):
    if instrument not in bucketed_returns:
        raise ValueError(f"No intra-month seasonality data available for {instrument!r}.")

    data = bucketed_returns[instrument]
    cumulative = data.fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=figsize)
    cumulative["first_7"].plot(ax=ax, label="First 7 Business Days", lw=1.6)
    cumulative["middle"].plot(ax=ax, label="Middle Days", lw=1.6)
    cumulative["last_7"].plot(ax=ax, label="Last 7 Business Days", lw=1.6)

    ax.set_title(f"{instrument} Intra-Month Return Seasonality")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax, cumulative


def save_intra_month_seasonality_pdf(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    output_path: str | Path,
    price_col: str = "mid",
    log_return: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bucketed_returns = calculate_intra_month_bucket_returns(
        quote_dict=quote_dict,
        price_col=price_col,
        log_return=log_return,
    )

    cumulative_results = {}
    with PdfPages(output_path) as pdf:
        for instrument in sorted(bucketed_returns):
            fig, _, cumulative = plot_intra_month_seasonality(
                bucketed_returns=bucketed_returns,
                instrument=instrument,
                figsize=figsize,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            cumulative_results[instrument] = cumulative

    return cumulative_results
