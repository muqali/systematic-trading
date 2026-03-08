import numpy as np

def mad_clip(df, z=5.0):
    '''MAD Winsorisation'''
    
    med = df.median()
    mad = (df - med).abs().median().replace(0, np.nan)
    rz = 0.6745 * (df - med) / mad

    lower = med - (z / 0.6745) * mad
    upper = med + (z / 0.6745) * mad

    return df.clip(lower=lower, upper=upper, axis=1)