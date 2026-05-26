import numpy as np
import pandas as pd


def attach_tz_column(
    data: pd.Series | pd.DataFrame,
    timezone: str | None = None,
    column_name: str = "tz",
) -> pd.DataFrame:
    """Return data with an intraday session label column.

    Session labels are based on the timestamp index. LDN starts at 07:30
    Europe/London local time and ends at 12:00 US/Eastern local time. NY is
    12:00 <= US/Eastern time < 17:00, and ASP is everything else.

    Naive timestamps are treated as US/Eastern for session labeling. Passing
    timezone also localizes/converts the returned index to that timezone.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex.")

    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    if timezone is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone)
        else:
            df.index = df.index.tz_convert(timezone)

    session_index = df.index
    if session_index.tz is None:
        session_index = session_index.tz_localize("US/Eastern")

    london_index = session_index.tz_convert("Europe/London")
    eastern_index = session_index.tz_convert("US/Eastern")

    london_hour = (
        london_index.hour
        + london_index.minute / 60
        + london_index.second / 3600
        + london_index.microsecond / 3_600_000_000
    )
    eastern_hour = (
        eastern_index.hour
        + eastern_index.minute / 60
        + eastern_index.second / 3600
        + eastern_index.microsecond / 3_600_000_000
    )

    df[column_name] = np.select(
        [
            (london_hour >= 7.5) & (eastern_hour < 12),
            (eastern_hour >= 12) & (eastern_hour < 17),
        ],
        ["LDN", "NYK"],
        default="ASP",
    )

    return df
