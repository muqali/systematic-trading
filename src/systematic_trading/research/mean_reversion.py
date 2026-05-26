from util.mis_util import calc_sharpe
from util.optimisation_util import sweep_parameters
import pandas as pd
import numpy as np
import statsmodels.api as sm


def regression_stats(x, y):
    sample = pd.concat({"x": x, "y": y}, axis=1).dropna()
    if len(sample) < 3 or sample["x"].var() == 0 or sample["y"].var() == 0:
        return np.nan, np.nan, len(sample)

    beta = sample["y"].cov(sample["x"]) / sample["x"].var()
    corr = sample["x"].corr(sample["y"])
    return beta, corr, len(sample)


def variance_ratio(
    ret_series: pd.Series, max_horizon: pd.Timedelta = pd.Timedelta(hours=1)
):
    freq = ret_series.index.diff().min()
    intervals = pd.timedelta_range(start=freq, end=max_horizon, freq=freq)
    variances = [ret_series.resample(interval).sum().var() for interval in intervals]

    df = pd.DataFrame(
        {
            "interval_seconds": intervals.total_seconds(),
            "variance/k": variances / intervals.total_seconds(),
        }
    ).set_index("interval_seconds")

    return df


def return_regression_heatmap(
    ret_series: pd.Series,
    past_periods: tuple[pd.Timedelta],
    future_periods: tuple[pd.Timedelta],
    zscore_window: pd.Timedelta,
    abs_z_buckets: tuple[tuple[float, float], ...] = (
        (0, 0.5),
        (0.5, 1),
        (1, 2),
        (2, np.inf),
    ),
):
    freq = ret_series.index.diff().min()
    results = []

    for p_timedelta in past_periods:
        for f_timedelta in future_periods:
            # Convert timedeltas into number of periods
            p = int(p_timedelta / freq)
            f = int(f_timedelta / freq)

            # Skip if periods are less than 1
            if p < 1 or f < 1:
                continue

            # Calculate rolling sum for past p periods
            return_past = ret_series.rolling(window=p).sum()

            abs_zscore_past = (
                (return_past - return_past.rolling(zscore_window).mean())
                / return_past.rolling(zscore_window).std()
            ).abs()

            # Calculate rolling sum for future f periods (shifted backwards)
            return_future = ret_series.rolling(window=f).sum().shift(-f)

            # Create DataFrame and drop NaN values
            df = pd.DataFrame(
                {
                    "return_past": return_past,
                    "return_future": return_future,
                    "abs_zscore_past": abs_zscore_past,
                }
            ).dropna()

            for lower, upper in abs_z_buckets:
                bucket_mask = (df["abs_zscore_past"] >= lower) & (
                    df["abs_zscore_past"] < upper
                )
                bucket_df = df[bucket_mask]

                # Skip if not enough data points
                if len(bucket_df) < 2:
                    continue

                # Calculate correlation
                correlation = bucket_df["return_past"].corr(bucket_df["return_future"])

                # Add constant for intercept
                X = sm.add_constant(bucket_df["return_past"], has_constant="add")
                y = bucket_df["return_future"]

                # Fit model
                model = sm.OLS(y, X).fit()
                abs_z_bucket = f"({lower},{upper})"

                results.append(
                    {
                        "name": ret_series.name,
                        "past_period": p_timedelta,
                        "future_period": f_timedelta,
                        "past_period_seconds": p_timedelta.total_seconds(),
                        "future_period_seconds": f_timedelta.total_seconds(),
                        "abs_z_bucket": abs_z_bucket,
                        "correlation": correlation,
                        "beta": model.params["return_past"],
                        "n_observations": len(bucket_df),
                        "r_squared": model.rsquared,
                        "alpha": model.params["const"],
                        "p_value": model.pvalues["return_past"],
                        "std_err": model.bse["return_past"],
                    }
                )

    return pd.DataFrame(results)


def rolling_return_regression(
    ret_series: pd.Series,
    past_horizon: pd.Timedelta = pd.Timedelta(minutes=60),
    future_horizon: pd.Timedelta = pd.Timedelta(minutes=5),
    lookback_window: pd.Timedelta = pd.Timedelta(days=5),
):
    freq = ret_series.index.diff().min()

    p = int(past_horizon / freq)
    f = int(future_horizon / freq)
    w = int(lookback_window / freq)

    # Calculate rolling sums
    return_past = ret_series.rolling(window=p).sum()
    return_future = ret_series.rolling(window=f).sum().shift(-f)

    df = pd.DataFrame({"x": return_past, "y": return_future}).dropna()

    x, y = df["x"], df["y"]

    # Rolling building blocks (all vectorised)
    roll = lambda s: s.rolling(window=w)

    sum_x = roll(x).sum()
    sum_y = roll(y).sum()
    sum_xx = roll(x**2).sum()
    sum_yy = roll(y**2).sum()
    sum_xy = roll(x * y).sum()

    # OLS closed-form
    denom = w * sum_xx - sum_x**2
    beta = (w * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / w

    # Predictions
    predictions = alpha.shift(f) + beta.shift(f) * x

    # Exact in-window RSS via OLS identify
    sum_resid_sq = sum_yy - beta * sum_xy - alpha * sum_y
    s2 = sum_resid_sq / (w - 2)
    se_beta = np.sqrt(s2 * w / denom)
    t_stat = beta / se_beta

    # Rolling correlation
    correlation = roll(x).corr(y)

    # Prediction z-scores
    roll_mean = roll(predictions).mean()
    roll_std = roll(predictions).std()
    prediction_zscore = (predictions - roll_mean) / roll_std

    results = pd.DataFrame(
        {
            "return_past": x,
            "return_future": y,
            "beta": beta,
            "alpha": alpha,
            "correlation": correlation,
            "t_stat": t_stat,
            "prediction": predictions,
            "prediction_zscore": prediction_zscore,
        }
    ).dropna()

    results.attrs["past_horizon"] = past_horizon
    results.attrs["future_horizon"] = future_horizon
    results.attrs["lookback_window"] = lookback_window

    return results


def apply_position(
    df: pd.DataFrame,
    tcost_bps: float = 1,
    margin_multiple: float = 2,
    adjust_to_target: bool = True,
    pred_zs_threshold: float = 2.0,
    delay_trade_on_signal: bool = False,
):
    df["tcost"] = tcost_bps / 10_000

    delay = 1 if delay_trade_on_signal else 0
    df["target_position"] = (
        df["prediction"]
        .shift(delay)
        .where(df["prediction_zscore"].shift(delay).abs() > pred_zs_threshold, 0)
    )

    margin = margin_multiple * df["tcost"]

    positions = np.empty(len(df))
    positions[:] = np.nan

    target = df["target_position"].values
    margin_vals = margin.values

    # Initialise first valid position
    first_valid = df["target_position"].first_valid_index()
    start = df.index.get_loc(first_valid)
    positions[start] = target[start]

    for i in range(start + 1, len(df)):
        current_pos = positions[i - 1]
        t = target[i]
        m = margin_vals[i]

        lower = t - m
        upper = t + m

        # Only move to target if outside the band around target
        if current_pos < lower:
            positions[i] = t if adjust_to_target else lower
        elif current_pos > upper:
            positions[i] = t if adjust_to_target else upper
        else:
            positions[i] = current_pos

    df["position"] = positions
    return df


def optimise_rolling_return_regression(
    ret_series,
    sort_key: str = "net_sharpe",
    grid: dict | None = None,
):
    if grid is None:
        grid = {
            "past_horizon": [
                pd.Timedelta(minutes=60),
                pd.Timedelta(minutes=120),
            ],
            "future_horizon": [
                pd.Timedelta(minutes=15),
                pd.Timedelta(minutes=30),
                pd.Timedelta(minutes=60),
            ],
            "lookback_window": [pd.Timedelta(days=5)],
            "tcost_bps": [1.0, 2.0],
            "margin_multiple": np.arange(0.0, 2.0, 0.5),
            "adjust_to_target": (True, False),
            "pred_zs_threshold": np.arange(0.0, 2.0, 0.5),
        }

    regression_cache: dict[
        tuple[pd.Timedelta, pd.Timedelta, pd.Timedelta], pd.DataFrame
    ] = {}

    def get_regression_df(
        past_horizon: pd.Timedelta,
        future_horizon: pd.Timedelta,
        lookback_window: pd.Timedelta,
    ) -> pd.DataFrame:
        key = (past_horizon, future_horizon, lookback_window)
        if key not in regression_cache:
            df = rolling_return_regression(
                ret_series, past_horizon, future_horizon, lookback_window
            )
            regression_cache[key] = df.join(ret_series.rename("ret"), how="left")

        return regression_cache[key]

    def objective_fn(
        past_horizon: pd.Timedelta,
        future_horizon: pd.Timedelta,
        lookback_window: pd.Timedelta,
        tcost_bps: float,
        margin_multiple: float,
        adjust_to_target: bool,
        pred_zs_threshold: float,
    ):
        df = get_regression_df(
            past_horizon,
            future_horizon,
            lookback_window,
        ).copy()
        df = apply_position(
            df, tcost_bps, margin_multiple, adjust_to_target, pred_zs_threshold
        )

        df["frictionless_pnl"] = df["target_position"].shift(1) * df["ret"]
        df["gross_pnl"] = df["position"].shift(1) * df["ret"]
        df["net_pnl"] = (
            df["gross_pnl"] - 0.5 * df["tcost"] * df["position"].diff().abs()
        )

        return {
            "net_sharpe": calc_sharpe(df["net_pnl"]),
            "gross_sharpe": calc_sharpe(df["gross_pnl"]),
            "frictionless_sharpe": calc_sharpe(df["frictionless_pnl"].dropna()),
        }

    sweep = sweep_parameters(grid, objective_fn)

    sweep = sweep.sort_values(sort_key, ascending=False)

    return sweep
