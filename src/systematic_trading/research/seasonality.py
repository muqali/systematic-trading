import math
from datetime import time
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _as_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    return pd.Timestamp(value).time()


def _get_price_series(data: pd.DataFrame | pd.Series, price_col: str) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()

    if price_col not in data.columns:
        raise ValueError(f"Expected column {price_col!r} in input data.")

    return data[price_col].dropna()


def _hour_window_labels() -> list[tuple[int, str, str]]:
    return [
        (hour, f"{hour:02d}:00", f"{(hour + 1) % 24:02d}:00")
        for hour in range(24)
    ]


def _calculate_window_return(
    price_series: pd.Series,
    start_time: time,
    end_time: time,
    log_return: bool,
) -> pd.Series:
    if start_time < end_time:
        intraday = price_series[
            (price_series.index.time >= start_time)
            & (price_series.index.time <= end_time)
        ]
        group_keys = intraday.index.normalize()
    else:
        intraday = price_series[
            (price_series.index.time >= start_time)
            | (price_series.index.time <= end_time)
        ]
        # Shift timestamps after midnight back one day so (23:00, 00:00]
        # is grouped with the preceding trade date.
        after_midnight = intraday.index.time <= end_time
        group_keys = intraday.index.normalize() - pd.to_timedelta(
            after_midnight.astype(int), unit="D"
        )
    if intraday.empty:
        return pd.Series(dtype=float)

    grouped = intraday.groupby(group_keys)
    window_prices = grouped.agg(["first", "last"]).dropna()
    if window_prices.empty:
        return pd.Series(dtype=float)

    if log_return:
        window_return = np.log(window_prices["last"] / window_prices["first"])
    else:
        window_return = window_prices["last"] / window_prices["first"] - 1.0

    window_return.name = price_series.name
    return window_return


def _calculate_daily_return(
    price_series: pd.Series,
    log_return: bool,
) -> pd.Series:
    if price_series.empty:
        return pd.Series(dtype=float)

    grouped = price_series.groupby(price_series.index.normalize())
    daily_prices = grouped.agg(["first", "last"]).dropna()
    if daily_prices.empty:
        return pd.Series(dtype=float)

    if log_return:
        daily_return = np.log(daily_prices["last"] / daily_prices["first"])
    else:
        daily_return = daily_prices["last"] / daily_prices["first"] - 1.0

    daily_return.name = price_series.name
    return daily_return


def calculate_intraday_window_returns(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    start_time: str | time = "07:00",
    end_time: str | time = "08:00",
    price_col: str = "mid",
    log_return: bool = True,
) -> pd.DataFrame:
    start_time = _as_time(start_time)
    end_time = _as_time(end_time)
    if start_time == end_time:
        raise ValueError("start_time and end_time must be different.")

    returns = {}
    for ccy_pair, data in quote_dict.items():
        price_series = _get_price_series(data, price_col)
        if price_series.empty:
            continue

        returns[ccy_pair] = _calculate_window_return(
            price_series=price_series,
            start_time=start_time,
            end_time=end_time,
            log_return=log_return,
        )

    if not returns:
        return pd.DataFrame()

    return pd.DataFrame(returns).sort_index()


def plot_intraday_return_seasonality(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    start_time: str | time = "07:00",
    end_time: str | time = "08:00",
    price_col: str = "mid",
    log_return: bool = True,
    overlay_all_hours: bool = True,
    figsize: tuple[float, float] | None = None,
):
    daily_returns = calculate_intraday_window_returns(
        quote_dict=quote_dict,
        start_time=start_time,
        end_time=end_time,
        price_col=price_col,
        log_return=log_return,
    )
    cumulative_returns = daily_returns.cumsum()

    if cumulative_returns.empty:
        raise ValueError("No intraday returns available for the requested window.")

    all_hours_cumulative_returns = {}
    if overlay_all_hours:
        for instrument in cumulative_returns.columns:
            price_series = _get_price_series(quote_dict[instrument], price_col)
            daily_full_return = _calculate_daily_return(
                price_series=price_series,
                log_return=log_return,
            )
            if daily_full_return.empty:
                continue
            all_hours_cumulative_returns[instrument] = daily_full_return.cumsum()

    instruments = list(cumulative_returns.columns)
    n_plots = len(instruments)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    if figsize is None:
        figsize = (5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for ax, instrument in zip(axes_flat, instruments, strict=False):
        cumulative_returns[instrument].dropna().plot(ax=ax, lw=1.5, label="Hour Window")
        if instrument in all_hours_cumulative_returns:
            all_hours_cumulative_returns[instrument].dropna().plot(
                ax=ax,
                lw=1.2,
                linestyle="--",
                color="black",
                label="All Hours",
            )
        ax.set_title(instrument)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Cumulative Intraday Return Seasonality ({start_time} to {end_time})",
        y=1.02,
    )
    fig.tight_layout()

    return fig, axes, daily_returns, cumulative_returns


def save_hourly_intraday_return_seasonality_pdf(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    output_path: str,
    price_col: str = "mid",
    log_return: bool = True,
    figsize: tuple[float, float] | None = None,
):
    hourly_returns = {}
    output_path = str(output_path)

    with PdfPages(output_path) as pdf:
        for hour in range(24):
            start_label = f"{hour:02d}:00"
            end_label = f"{(hour + 1) % 24:02d}:00"

            try:
                fig, axes, daily_returns, cumulative_returns = plot_intraday_return_seasonality(
                    quote_dict=quote_dict,
                    start_time=start_label,
                    end_time=end_label,
                    price_col=price_col,
                    log_return=log_return,
                    figsize=figsize,
                )
            except ValueError:
                fig, ax = plt.subplots(figsize=figsize or (10, 4))
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"No intraday returns available for {start_label} to {end_label}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                fig.suptitle(
                    f"Cumulative Intraday Return Seasonality ({start_label} to {end_label})"
                )
                daily_returns = pd.DataFrame()
                cumulative_returns = pd.DataFrame()

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            hourly_returns[f"{start_label}-{end_label}"] = {
                "daily_returns": daily_returns,
                "cumulative_returns": cumulative_returns,
            }

    return hourly_returns


def save_pair_intraday_return_seasonality_pdf(
    hourly_results: dict[str, dict[str, pd.DataFrame]],
    output_path: str,
    figsize: tuple[float, float] = (14, 18),
):
    output_path = str(output_path)

    instruments = sorted(
        {
            instrument
            for result in hourly_results.values()
            for instrument in result.get("cumulative_returns", pd.DataFrame()).columns
        }
    )

    with PdfPages(output_path) as pdf:
        for instrument in instruments:
            fig, axes = plt.subplots(6, 4, figsize=figsize, squeeze=False)
            axes_flat = axes.flatten()
            window_items = list(hourly_results.items())
            all_hours_daily = []

            for result in hourly_results.values():
                daily_returns = result.get("daily_returns", pd.DataFrame())
                if instrument in daily_returns.columns:
                    all_hours_daily.append(daily_returns[instrument])

            if all_hours_daily:
                all_hours_benchmark = (
                    pd.concat(all_hours_daily, axis=1)
                    .sort_index()
                    .sum(axis=1, min_count=1)
                    .groupby(level=0)
                    .sum(min_count=1)
                    .sort_index()
                    .cumsum()
                )
            else:
                all_hours_benchmark = pd.Series(dtype=float)

            for ax, (window_label, result) in zip(axes_flat, window_items, strict=False):
                cumulative_returns = result.get("cumulative_returns", pd.DataFrame())

                if instrument not in cumulative_returns.columns:
                    ax.set_visible(False)
                    continue

                hour_series = (
                    cumulative_returns[instrument]
                    .dropna()
                    .groupby(level=0)
                    .last()
                    .sort_index()
                )
                if hour_series.empty:
                    ax.set_visible(False)
                    continue

                hour_series.plot(ax=ax, lw=1.2, alpha=0.9, label="Hour Window")

                if not all_hours_benchmark.empty:
                    all_hours_benchmark.dropna().plot(
                        ax=ax,
                        lw=1.1,
                        color="black",
                        linestyle="--",
                        label="All Hours",
                    )

                ax.set_title(window_label, fontsize=10)
                ax.set_xlabel("Date")
                ax.set_ylabel("Cum Return")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=7)

            for ax in axes_flat[len(window_items):]:
                ax.set_visible(False)

            fig.suptitle(f"{instrument} Intraday Return Seasonality", y=0.995)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def calculate_average_return_by_hour(
    quote_dict: dict[str, pd.DataFrame | pd.Series],
    price_col: str = "mid",
    log_return: bool = True,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    rows = []

    for ccy_pair, data in quote_dict.items():
        price_series = _get_price_series(data, price_col)
        if price_series.empty:
            continue

        for hour, start_label, end_label in _hour_window_labels():
            hourly_returns = _calculate_window_return(
                price_series=price_series,
                start_time=_as_time(start_label),
                end_time=_as_time(end_label),
                log_return=log_return,
            ).dropna()

            n_obs = len(hourly_returns)
            if n_obs == 0:
                mean_return = np.nan
                std_return = np.nan
                std_error = np.nan
                conf_low = np.nan
                conf_high = np.nan
            else:
                mean_return = float(hourly_returns.mean())
                if n_obs >= 2:
                    std_return = float(hourly_returns.std(ddof=1))
                    std_error = std_return / np.sqrt(n_obs)
                    band = z_score * std_error
                    conf_low = mean_return - band
                    conf_high = mean_return + band
                else:
                    std_return = np.nan
                    std_error = np.nan
                    conf_low = np.nan
                    conf_high = np.nan

            rows.append(
                {
                    "ccy_pair": ccy_pair,
                    "hour": hour,
                    "start_time": start_label,
                    "end_time": end_label,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "std_error": std_error,
                    "n_obs": n_obs,
                    "conf_low": conf_low,
                    "conf_high": conf_high,
                }
            )

    return pd.DataFrame(rows).sort_values(["ccy_pair", "hour"]).reset_index(drop=True)


def plot_hourly_average_return_bands(
    hourly_stats: pd.DataFrame,
    instrument: str,
    figsize: tuple[float, float] = (10, 5),
):
    pair_stats = (
        hourly_stats.loc[hourly_stats["ccy_pair"] == instrument]
        .sort_values("hour")
        .reset_index(drop=True)
    )
    if pair_stats.empty:
        raise ValueError(f"No hourly statistics available for {instrument!r}.")

    x = np.arange(len(pair_stats))
    mean = pair_stats["mean_return"].to_numpy()
    conf_low = pair_stats["conf_low"].to_numpy()
    conf_high = pair_stats["conf_high"].to_numpy()
    end_labels = pair_stats["end_time"].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    valid_band = np.isfinite(conf_low) & np.isfinite(conf_high)
    if valid_band.any():
        ax.fill_between(
            x[valid_band],
            conf_low[valid_band],
            conf_high[valid_band],
            color="tab:blue",
            alpha=0.18,
            label="Confidence Band",
        )
        ax.vlines(
            x[valid_band],
            conf_low[valid_band],
            conf_high[valid_band],
            color="tab:blue",
            alpha=0.6,
            linewidth=1.0,
        )

    ax.plot(
        x,
        mean,
        color="tab:blue",
        marker="o",
        linewidth=1.5,
        label="Average Return",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(end_labels, rotation=45)
    ax.set_xlabel("Hour Ending")
    ax.set_ylabel("Average Return")
    ax.set_title(f"{instrument} Average Return by Hour")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax, pair_stats


def save_hourly_average_return_bands_pdf(
    hourly_stats: pd.DataFrame,
    output_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> list[str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instruments = sorted(hourly_stats["ccy_pair"].dropna().unique())
    with PdfPages(output_path) as pdf:
        for instrument in instruments:
            fig, _, _ = plot_hourly_average_return_bands(
                hourly_stats=hourly_stats,
                instrument=instrument,
                figsize=figsize,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return instruments
