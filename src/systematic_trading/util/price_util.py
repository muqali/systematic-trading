import pandas as pd


def _infer_even_frequency(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return None
    if (diffs != diffs.iloc[0]).any():
        raise ValueError("ret_series must be evenly spaced.")

    freq = diffs.iloc[0]
    if freq <= pd.Timedelta(0):
        raise ValueError("ret_series index must be strictly increasing.")
    return freq


def attach_volatility(
    ret_series: pd.Series, half_life: pd.Timedelta, normalise_vol: bool = False
) -> pd.DataFrame:
    """
    ret_series is an evenly spaced time series with timestamp index and price return of
    an asset. At row t, value represents return from t-1 to t.
    Calculate volatility as EMA of the past absolute returns with half_life
    Attach the calculated volatility as a new column
    """
    if not isinstance(ret_series.index, pd.DatetimeIndex):
        raise TypeError("ret_series must have a DatetimeIndex.")
    if half_life <= pd.Timedelta(0):
        raise ValueError("half_life must be positive.")

    ret_series = ret_series.sort_index()
    df = ret_series.to_frame(name=ret_series.name or "ret")

    freq = _infer_even_frequency(ret_series.index)
    if freq is None:
        df["volatility"] = float("nan")
        return df

    half_life_periods = half_life / freq
    normalisation_factor = (
        (freq / pd.Timedelta(days=1)) ** 0.5 if normalise_vol else 1.0
    )
    df["volatility"] = (
        ret_series.abs().shift(1).ewm(halflife=half_life_periods, adjust=False).mean()
        / normalisation_factor
    )

    return df


def attach_trendiness(
    ret_series: pd.Series, past_horizon: pd.Timedelta
) -> pd.DataFrame:
    """
    ret_series is an evenly spaced time series with timestamp index and price return of
    an asset. At row t, value represents return from t-1 to t.
    Calculate trendiness as abs(sum of returns) divided by sum(abs(returns)) over the rolling past horizon
    Attach result as new column
    """
    if not isinstance(ret_series.index, pd.DatetimeIndex):
        raise TypeError("ret_series must have a DatetimeIndex.")
    if past_horizon <= pd.Timedelta(0):
        raise ValueError("past_horizon must be positive.")

    ret_series = ret_series.sort_index()
    df = ret_series.to_frame(name=ret_series.name or "ret")

    freq = _infer_even_frequency(ret_series.index)
    if freq is None:
        df["trendiness"] = float("nan")
        return df

    window = int(past_horizon / freq)
    if window < 1:
        raise ValueError("past_horizon must be at least one period.")

    net_return = ret_series.rolling(window=window).sum().abs()
    gross_return = ret_series.abs().rolling(window=window).sum()
    df["trendiness"] = net_return / gross_return.replace(0.0, float("nan"))

    return df
