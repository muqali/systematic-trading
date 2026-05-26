from research.mean_reversion import return_regression_heatmap
import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_WEIGHTS = {
    "USDSGD": 0.1987,
    "EURUSD": 0.1503,
    "USDCNH": 0.1476,
    "USDMYR": 0.1162,
    "USDJPY": 0.0938,
    "AUDUSD": 0.065,
    "USDINR": 0.0533,
    "USDKRW": 0.046,
    "USDTHB": 0.1083,
    "USDIDR": 0.0313,
    "USDTWD": 0.0244,
    "GBPUSD": 0.0182,
    "USDHKD": 0.016,
}

INDEX_CCY_TO_PAIR_MAP = {
    "USD": "USDSGD",
    "EUR": "EURUSD",
    "CNH": "USDCNH",
    "MYR": "USDMYR",
    "JPY": "USDJPY",
    "AUD": "AUDUSD",
    "INR": "USDINR",
    "KRW": "USDKRW",
    "THB": "USDTHB",
    "IDR": "USDIDR",
    "TWD": "USDTWD",
    "GBP": "GBPUSD",
    "HKD": "USDHKD",
}


def compute_mid_returns(price_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mids = {pair: df["mid"] for pair, df in price_dict.items() if "mid" in df.columns}
    if not mids:
        raise ValueError(
            "fx_price_dict must contain at least one DataFrame with a mid column."
        )

    logp = np.log(pd.DataFrame(mids).sort_index())
    return logp.diff()


def build_rets_vs_sgd(price_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build log returns for SGD against NEER basket currencies.

    The input prices are FX pairs quoted against USD and must include USDSGD.
    Output currency columns use SGD as the base currency, so column ``X`` is
    the log return of SGD/X. The ``index`` column is the normalised weighted
    average of the available pairs in ``DEFAULT_WEIGHTS`` and is NaN whenever
    any included component return is NaN.
    """
    rets = compute_mid_returns(price_dict).copy()
    if "USDSGD" not in rets.columns:
        raise ValueError("USDSGD is required to build SGD NEER returns.")

    available_weighted_pairs = [
        pair for pair in DEFAULT_WEIGHTS if pair in rets.columns
    ]
    weights = normalised_weights(available_weighted_pairs)

    usd_ccy_rets: dict[str, pd.Series] = {"USD": -rets["USDSGD"]}
    index_ret = pd.Series(0.0, index=rets.index, dtype=float)

    for pair, weight in weights.items():
        if pair == "USDSGD":
            sgd_ccy_ret = -rets[pair]
        elif pair.startswith("USD"):
            ccy = pair[3:]
            usd_ccy_rets[ccy] = rets[pair]
            sgd_ccy_ret = rets[pair] - rets["USDSGD"]
        else:
            ccy = pair[:3]
            usd_ccy_rets[ccy] = -rets[pair]
            sgd_ccy_ret = -rets[pair] - rets["USDSGD"]

        index_ret = index_ret + weight * sgd_ccy_ret

    rets_vs_sgd = pd.DataFrame(index=rets.index)
    rets_vs_sgd["index"] = index_ret
    for ccy in usd_ccy_rets:
        if ccy == "USD":
            rets_vs_sgd[ccy] = usd_ccy_rets[ccy]
        else:
            rets_vs_sgd[ccy] = usd_ccy_rets[ccy] - rets["USDSGD"]

    return rets_vs_sgd


def normalised_weights(pairs: list[str]) -> dict[str, float]:
    raw_weights = {pair: DEFAULT_WEIGHTS.get(pair, 0) for pair in pairs}
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        raise ValueError("At least one positive NEER weight is required.")
    return {pair: weight / total_weight for pair, weight in raw_weights.items()}


def rolling_panel_regression(
    index_ret_series: pd.Series,
    currency_ret_series: pd.Series,
    past_horizon: pd.Timedelta = pd.Timedelta(minutes=60),
    future_horizon: pd.Timedelta = pd.Timedelta(minutes=5),
    lookback_window: pd.Timedelta = pd.Timedelta(days=5),
    first_stage_beta_mode: str = "weight",
    pair_weights: dict[str, float] = DEFAULT_WEIGHTS,
):
    if first_stage_beta_mode not in ("regression", "weight"):
        raise ValueError(
            "first_stage_beta_mode must be either 'regression' or 'weight'."
        )

    ccy = currency_ret_series.name

    freq = index_ret_series.index.diff().min()

    p = int(past_horizon / freq)
    f = int(future_horizon / freq)
    w = int(lookback_window / freq)

    index_past = index_ret_series.rolling(window=p).sum()

    ccy_past = currency_ret_series.rolling(window=p).sum()
    ccy_future = currency_ret_series.rolling(window=f).sum().shift(-f)

    df = pd.DataFrame(
        {
            "index_return_past": index_past,
            "ccy_return_past": ccy_past,
            "ccy_return_future": ccy_future,
        }
    ).dropna()

    x = df["index_return_past"]
    z = df["ccy_return_past"]
    y = df["ccy_return_future"]

    roll = lambda s: s.rolling(window=w)

    sum_x = roll(x).sum()
    sum_z = roll(z).sum()
    sum_y = roll(y).sum()
    sum_xx = roll(x**2).sum()
    sum_zz = roll(z**2).sum()
    sum_yy = roll(y**2).sum()
    sum_xz = roll(x * z).sum()
    sum_xy = roll(x * y).sum()
    sum_zy = roll(z * y).sum()

    # First stage: ccy past return ~ beta_c * index past return + const.
    s_xx = sum_xx - sum_x**2 / w
    s_zz = sum_zz - sum_z**2 / w
    s_yy = sum_yy - sum_y**2 / w
    s_xz = sum_xz - sum_x * sum_z / w
    s_xy = sum_xy - sum_x * sum_y / w
    s_zy = sum_zy - sum_z * sum_y / w

    if first_stage_beta_mode == "regression":
        first_stage_beta = s_xz / s_xx
        first_stage_alpha = (sum_z - first_stage_beta * sum_x) / w
        residual = z - first_stage_alpha - first_stage_beta * x

        # The full in-window first-stage residual vector is orthogonal to the
        # intercept and x, so this two-factor OLS has simple closed forms.
        residual_sum_sq = s_zz - first_stage_beta * s_xz
        residual_sum_y = s_zy - first_stage_beta * s_xy

        residual_beta = residual_sum_y / residual_sum_sq
        index_beta = s_xy / s_xx
        alpha = (sum_y - index_beta * sum_x) / w
        sum_resid_sq = s_yy - residual_beta * residual_sum_y - index_beta * s_xy
        se_residual_beta = np.sqrt((sum_resid_sq / (w - 3)) / residual_sum_sq)
        correlation = residual_sum_y / np.sqrt(residual_sum_sq * s_yy)
    else:
        if ccy is None:
            raise ValueError(
                "currency_ret_series.name must be set when "
                "first_stage_beta_mode='weight'."
            )
        if ccy not in INDEX_CCY_TO_PAIR_MAP:
            raise ValueError(f"No index pair mapping found for currency {ccy!r}.")

        pair = INDEX_CCY_TO_PAIR_MAP[ccy]
        if pair not in pair_weights:
            raise ValueError(f"No pair weight found for {pair!r}.")

        first_stage_beta = pd.Series(float(pair_weights[pair]), index=x.index)
        first_stage_alpha = pd.Series(0.0, index=x.index)
        residual = z - first_stage_beta * x

        sum_r = roll(residual).sum()
        sum_rr = roll(residual**2).sum()
        sum_rx = roll(residual * x).sum()
        sum_ry = roll(residual * y).sum()

        s_rr = sum_rr - sum_r**2 / w
        s_rx = sum_rx - sum_r * sum_x / w
        s_ry = sum_ry - sum_r * sum_y / w

        denom = s_rr * s_xx - s_rx**2
        residual_beta = (s_ry * s_xx - s_xy * s_rx) / denom
        index_beta = (s_xy * s_rr - s_ry * s_rx) / denom
        alpha = (sum_y - residual_beta * sum_r - index_beta * sum_x) / w

        sum_resid_sq = s_yy - residual_beta * s_ry - index_beta * s_xy
        s2 = sum_resid_sq / (w - 3)
        se_residual_beta = np.sqrt(s2 * s_xx / denom)
        correlation = s_ry / np.sqrt(s_rr * s_yy)

    prediction = (
        alpha.shift(f) + residual_beta.shift(f) * residual + index_beta.shift(f) * x
    )

    t_stat = residual_beta / se_residual_beta

    roll_prediction = roll(prediction)
    prediction_zscore = (prediction - roll_prediction.mean()) / roll_prediction.std()

    results = (
        pd.DataFrame(
            {
                "return_past": residual,
                "return_future": y,
                "index_return_past": x,
                "ccy_return_past": z,
                "ccy_return_future": y,
                "first_stage_beta": first_stage_beta,
                "first_stage_alpha": first_stage_alpha,
                "residual": residual,
                "beta": residual_beta,
                "residual_beta": residual_beta,
                "index_beta": index_beta,
                "alpha": alpha,
                "correlation": correlation,
                "t_stat": t_stat,
                "prediction": prediction,
                "prediction_zscore": prediction_zscore,
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    results.attrs["past_horizon"] = past_horizon
    results.attrs["future_horizon"] = future_horizon
    results.attrs["lookback_window"] = lookback_window
    results.attrs["first_stage_beta_mode"] = first_stage_beta_mode
    results.attrs["ccy"] = ccy

    return results


def panel_regression_heatmap(
    index_ret_series: pd.Series,
    currency_ret_series: pd.Series,
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
    freq = index_ret_series.index.diff().min()
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
            index_return_past = index_ret_series.rolling(window=p).sum()
            ccy_return_past = currency_ret_series.rolling(window=p).sum()
            ccy_relative_return_past = ccy_return_past - index_return_past

            abs_zscore_ccy_relative_past = (
                (
                    ccy_relative_return_past
                    - ccy_relative_return_past.rolling(zscore_window).mean()
                )
                / ccy_relative_return_past.rolling(zscore_window).std()
            ).abs()

            # Calculate rolling sum for future f periods (shifted backwards)
            ccy_return_future = currency_ret_series.rolling(window=f).sum().shift(-f)

            # Create DataFrame and drop NaN values
            df = pd.DataFrame(
                {
                    "index_return_past": index_return_past,
                    "ccy_return_past": ccy_return_past,
                    "ccy_relative_return_past": ccy_relative_return_past,
                    "ccy_return_future": ccy_return_future,
                    "abs_zscore_ccy_relative_past": abs_zscore_ccy_relative_past,
                }
            ).dropna()

            for lower, upper in abs_z_buckets:
                bucket_mask = (df["abs_zscore_ccy_relative_past"] >= lower) & (
                    df["abs_zscore_ccy_relative_past"] < upper
                )
                bucket_df = df[bucket_mask]

                # Skip if not enough data points
                if len(bucket_df) < 3:
                    continue

                # Calculate marginal correlations
                index_correlation = bucket_df["index_return_past"].corr(
                    bucket_df["ccy_return_future"]
                )
                relative_correlation = bucket_df["ccy_relative_return_past"].corr(
                    bucket_df["ccy_return_future"]
                )

                # Add constant for intercept
                X = sm.add_constant(
                    bucket_df[["index_return_past", "ccy_relative_return_past"]],
                    has_constant="add",
                )
                y = bucket_df["ccy_return_future"]

                # Fit model
                model = sm.OLS(y, X).fit()
                abs_z_bucket = f"{lower}-{upper}"

                results.append(
                    {
                        "name": currency_ret_series.name,
                        "past_period": p_timedelta,
                        "future_period": f_timedelta,
                        "past_period_seconds": p_timedelta.total_seconds(),
                        "future_period_seconds": f_timedelta.total_seconds(),
                        "abs_z_bucket": abs_z_bucket,
                        "index_correlation": index_correlation,
                        "relative_correlation": relative_correlation,
                        "relative_beta": model.params["ccy_relative_return_past"],
                        "index_beta": model.params["index_return_past"],
                        "n_observations": len(bucket_df),
                        "r_squared": model.rsquared,
                        "alpha": model.params["const"],
                        "relative_p_value": model.pvalues["ccy_relative_return_past"],
                        "index_p_value": model.pvalues["index_return_past"],
                        "relative_std_err": model.bse["ccy_relative_return_past"],
                        "index_std_err": model.bse["index_return_past"],
                    }
                )

    return pd.DataFrame(results)
