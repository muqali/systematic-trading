"""Daily-clock and continuously rolling SGD basket experiments.

Rolling horizons count scheduled market minutes, including missing quotes.
The weekend closure is excluded; outages are never compressed. Financing is
excluded, consistently with the user's requested research assumption.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def clock_marks(data, hour):
    """Same estimator as the multi-day study, at a specified New York hour."""
    if not 0 <= hour <= 16:
        raise ValueError('Use a weekday hour between 00:00 and 16:00 New York.')
    local = data.level.index.tz_convert('America/New_York')
    dates = pd.date_range(local.min().normalize(), local.max().normalize(), freq='B')
    grid = dates + pd.Timedelta(hours=hour)
    out = pd.DataFrame({'level': data.level.reindex(grid),
                        'spread': data.full_spread.reindex(grid)})
    vol = out.level.diff().ewm(halflife=20, min_periods=40).std().shift(1)
    out['vol'] = vol.clip(lower=vol.rolling(252, min_periods=60).median()*.5)
    return out


def market_grid(index):
    """Minute grid using the source loader's Sun 18:00--Fri 17:00 NY week."""
    local = index.tz_convert('America/New_York')
    full = pd.date_range(local.min().floor('min'), local.max().ceil('min'), freq='min')
    weekday, hour = full.dayofweek, full.hour
    return full[(weekday < 4) | ((weekday == 4) & (hour < 17)) |
                ((weekday == 6) & (hour >= 18))]


def rolling_marks(data):
    """Minute marks and prior empirical 24-market-hour return volatility.

    Use the dispersion of actual long returns, rather than scaling minute
    volatility by sqrt(horizon). Overlapping observations provide a smoothly
    updated estimator, not additional independent days of evidence.
    """
    grid = market_grid(data.level.index)
    out = pd.DataFrame({'level': data.level.reindex(grid),
                        'spread': data.full_spread.reindex(grid)})
    daily_move = out.level.diff(24*60)
    out['vol'] = daily_move.ewm(halflife=20*1440, min_periods=40*1440).std().shift(1)
    return out


def rolling_features(marks, kind, horizon_hours):
    n = int(horizon_hours*60)
    if n < 1 or n != horizon_hours*60:
        raise ValueError('Horizon must be a positive whole number of minutes.')
    if kind == 'return':
        move = marks.level.diff(n)
        mean = move.ewm(halflife=63*1440, min_periods=40*1440).mean().shift(1)
        scale = move.ewm(halflife=63*1440, min_periods=40*1440).std().shift(1)
        deviation = move-mean
    elif kind == 'equilibrium':
        mean = marks.level.ewm(halflife=n, min_periods=max(20*1440, 2*n)).mean().shift(1)
        deviation = marks.level-mean
        scale = deviation.ewm(halflife=63*1440, min_periods=40*1440).std().shift(1)
    else:
        raise ValueError(f'Unknown rolling signal kind: {kind}')
    return pd.DataFrame({'z': deviation/scale.replace(0, np.nan),
                         'edge': -.5*deviation}, index=marks.index)


@dataclass(frozen=True)
class RollingPolicy:
    name: str
    kind: str
    horizon_hours: int
    entry_z: float = 2.
    review_hour: int | None = None
    exit_z: float = .25
    cost_hurdle: float = 1.5
    vol_target: float = .05
    max_units: float = 3.
    max_holding_hours: float | str | None = None


def rolling_signal(marks, features, policy):
    """Sparse target changes; every-minute review unless review_hour is set.

    Entry size is held constant. All positions can cross daily/weekend closes.
    Maximum holding time defaults to twice the lookback in scheduled market
    minutes. max_holding_hours overrides this; 'unlimited' disables time exits.
    A time exit during an outage is requested immediately and must reach an
    available quote before a new entry is allowed. No same-minute re-entry.
    """
    index = marks.index
    risk = (policy.vol_target/(marks.vol*np.sqrt(252))).clip(upper=policy.max_units)
    z, edge, spread, units = (x.to_numpy() for x in
                              [features.z, features.edge, marks.spread, risk])
    valid = marks.level.notna().to_numpy() & marks.spread.notna().to_numpy()
    usable = np.isfinite(z) & np.isfinite(edge) & np.isfinite(units) & valid
    review = np.ones(len(index), dtype=bool)
    if policy.review_hour is not None:
        local = index.tz_convert('America/New_York')
        review = (local.hour == policy.review_hour) & (local.minute == 0)
    entry = usable & review & (np.abs(z) >= policy.entry_z) & (np.abs(edge) > policy.cost_hurdle*spread)
    changes, values = [0], [0.]
    q, entered, pending_close = 0., 0, False
    holding = policy.max_holding_hours
    if holding == 'unlimited':
        limit = np.inf
    else:
        hours = 2*policy.horizon_hours if holding is None else float(holding)
        if not np.isfinite(hours) or hours <= 0 or hours*60 != int(hours*60):
            raise ValueError('max_holding_hours must span positive whole minutes, or be unlimited.')
        limit = int(hours*60)
    for i in range(len(index)):
        previous = q
        if q and i-entered >= limit:
            q, pending_close = 0., not valid[i]
        elif pending_close:
            if valid[i]:
                pending_close = False
        elif q:
            if usable[i] and review[i] and q*z[i] >= -abs(q)*policy.exit_z:
                q = 0.
        elif entry[i]:
            q, entered = np.sign(edge[i])*units[i], i
        if q != previous:
            changes.append(i)
            values.append(q)
    # The first input timestamp can also be a valid entry in small test samples.
    return pd.Series(values, index=index[changes], name=policy.name).groupby(level=0).last()
