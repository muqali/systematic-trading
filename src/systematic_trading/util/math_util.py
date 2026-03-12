import numpy as np
import pandas as pd


def mad_clip(df, z=5.0):
    """MAD winsorization, applied independently to each column."""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    med = df.median(axis=0)
    mad = (df.sub(med, axis=1)).abs().median(axis=0).replace(0, np.nan)

    lower = med - (z / 0.6745) * mad
    upper = med + (z / 0.6745) * mad

    return df.clip(lower=lower, upper=upper, axis=1)
