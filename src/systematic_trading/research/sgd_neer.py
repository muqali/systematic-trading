import numpy as np
import pandas as pd


def regression_stats(x, y):
    sample = pd.concat({"x": x, "y": y}, axis=1).dropna()
    if len(sample) < 3 or sample["x"].var() == 0 or sample["y"].var() == 0:
        return np.nan, np.nan, len(sample)

    beta = sample["y"].cov(sample["x"]) / sample["x"].var()
    corr = sample["x"].corr(sample["y"])
    return beta, corr, len(sample)


def sharpe(pnl_series: pd.Series):
    daily_pnl_series = pnl_series.resample("1D").sum()

    return daily_pnl_series.mean() / daily_pnl_series.std() * np.sqrt(252)


def rolling_panel_regression(
    index_ret_series: pd.Series,
    currency_ret_series: pd.Series,
    past_horizon: pd.Timedelta = pd.Timedelta(minutes=60),
    future_horizon: pd.Timedelta = pd.Timedelta(minutes=5),
    lookback_window: pd.Timedelta = pd.Timedelta(days=5),
):
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

    first_stage_beta = s_xz / s_xx
    first_stage_alpha = (sum_z - first_stage_beta * sum_x) / w
    residual = z - first_stage_alpha - first_stage_beta * x

    # In each window the first-stage residual is orthogonal to the intercept
    # and index return, so the second-stage coefficients have simple forms.
    residual_sum_sq = s_zz - first_stage_beta * s_xz
    residual_sum_y = s_zy - first_stage_beta * s_xy

    residual_beta = residual_sum_y / residual_sum_sq
    index_beta = s_xy / s_xx
    alpha = (sum_y - index_beta * sum_x) / w

    prediction = alpha + residual_beta * residual + index_beta * x

    fitted_sum_y = residual_beta * residual_sum_y + index_beta * s_xy
    sum_resid_sq = s_yy - fitted_sum_y
    s2 = sum_resid_sq / (w - 3)
    se_residual_beta = np.sqrt(s2 / residual_sum_sq)
    t_stat = residual_beta / se_residual_beta

    correlation = residual_sum_y / np.sqrt(residual_sum_sq * s_yy)

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

    return results
