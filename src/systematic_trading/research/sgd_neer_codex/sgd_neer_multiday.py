"""Business-day signals with minute execution and overnight holding.

All daily decisions are at 16:00 New York. Missing decision marks stay missing;
business-day horizons never compress missing dates. Existing positions stay live.
Forward-point adjustments are excluded at the user's request.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyPolicy:
    name: str
    kind: str
    horizon_days: int
    entry_z: float = 2.
    exit_z: float = .25
    cost_hurdle: float = 1.5
    max_hold_days: int = 10
    vol_target: float = .05
    max_units: float = 3.


def daily_marks(data):
    """Exact 16:00 common quotes; no interpolation or nearest future quote."""
    ny = data.mids.index.tz_convert('America/New_York')
    dates = pd.date_range(ny.min().normalize(), ny.max().normalize(), freq='B')
    grid = dates + pd.Timedelta(hours=16)
    out = pd.DataFrame({'level': data.level.reindex(grid),
                        'spread': data.full_spread.reindex(grid)})
    ret = out.level.diff()
    vol = ret.ewm(halflife=20, min_periods=40).std().shift(1)
    out['vol'] = vol.clip(lower=vol.rolling(252, min_periods=60).median()*.5)
    return out


def daily_regression(level, horizon, training_days=252):
    """Weekly slope of future H-day returns on past H-day returns, beta <= 0.

    Sample endpoints are fixed business dates, including missing dates. Only
    labels whose actual endpoint precedes the fit are available for estimation.
    No fitted intercept is traded. Input/output are log returns, not z-scores.
    """
    x = level.diff(horizon)
    y = level.shift(-horizon)-level
    ends = level.index.to_series().shift(-horizon)
    sample = pd.DataFrame({'x': x, 'y': y, 'end': ends}).dropna()
    forecast = pd.Series(np.nan, index=level.index, name='forecast')
    fits = []
    weeks = pd.date_range(level.index.min().normalize(), level.index.max()+pd.Timedelta(days=8), freq='W-SAT')
    for start, end in zip(weeks[:-1], weeks[1:]):
        right = level.index.searchsorted(start)
        if right < training_days:
            continue
        left_time = level.index[max(0, right-training_days)]
        train = sample.loc[(sample.index >= left_time) & (sample.end < start)]
        if len(train) < 100 or train.x.var() <= 0:
            continue
        beta = min(0., train.x.cov(train.y)/train.x.var())
        mask = (level.index >= start) & (level.index < end)
        forecast.loc[mask] = beta*x.loc[mask]
        fits.append({'fit_ts': start, 'last_label_end': train.end.max(),
                     'n': len(train), 'beta': beta, 'horizon_days': horizon})
    return forecast, pd.DataFrame(fits)


def daily_features(marks, kind, horizon):
    level = marks.level
    fits = pd.DataFrame()
    if kind == 'return':
        move = level.diff(horizon)
        # Remove the prior local drift before interpreting an unusual move.
        center = move.ewm(halflife=63, min_periods=40).mean().shift(1)
        scale = move.ewm(halflife=63, min_periods=40).std().shift(1)
        deviation = move-center
        z = deviation/scale.replace(0, np.nan)
        edge = -.5*deviation
    elif kind == 'equilibrium':
        mean = level.ewm(halflife=horizon, min_periods=max(20, 2*horizon)).mean().shift(1)
        deviation = level-mean
        scale = deviation.ewm(halflife=63, min_periods=40).std().shift(1)
        z = deviation/scale.replace(0, np.nan)
        edge = -.5*deviation
    elif kind == 'forecast':
        edge, fits = daily_regression(level, horizon)
        center = edge.rolling(126, min_periods=60).mean().shift(1)
        scale = edge.rolling(126, min_periods=60).std().shift(1)
        z = (edge-center)/scale.replace(0, np.nan)
    else:
        raise ValueError(f'Unknown signal kind: {kind}')
    return pd.DataFrame({'z': z, 'edge': edge}), fits


def daily_signal(marks, features, policy):
    """Enter on |z| >= threshold AND edge > 1.5 spreads; hold through rolls.

    Position size is fixed at entry. Exit near/crossing equilibrium, on forecast
    reversal, or at the maximum business-day age. Missing decision quotes defer
    price-dependent decisions. A known time limit can still request an exit,
    executed at the next available minute quote with intervening PnL retained.
    No same-decision re-entry.
    """
    risk = (policy.vol_target/(marks.vol*np.sqrt(252))).clip(upper=policy.max_units)
    z, edge = features.z.to_numpy(), features.edge.to_numpy()
    valid = marks.level.notna().to_numpy() & marks.spread.notna().to_numpy()
    spread, units = marks.spread.to_numpy(), risk.to_numpy()
    signal = np.zeros(len(marks))
    q, age = 0., 0
    for i in range(len(marks)):
        if q:
            age += 1
        usable = np.isfinite(z[i]) and np.isfinite(edge[i]) and np.isfinite(units[i])
        if q and age >= policy.max_hold_days:
            q, age = 0., 0
        elif valid[i]:
            if q:
                near_mean = (q*z[i] >= -abs(q)*policy.exit_z if policy.kind != 'forecast'
                             else q*edge[i] <= 0 or abs(z[i]) <= policy.exit_z)
                if usable and near_mean:
                    q, age = 0., 0
            elif usable and abs(z[i]) >= policy.entry_z and abs(edge[i]) > policy.cost_hurdle*spread[i]:
                q, age = np.sign(edge[i])*units[i], 0
        signal[i] = q
    return pd.Series(signal, index=marks.index, name=policy.name)


class MinuteAccountant:
    """Exact scalar-basket spread accounting, aggregated to daily PnL cheaply.

    Costs are calculated only at position changes, with the same directional
    log half-spread formula as the general five-leg engine. All minute/gap
    returns are earned by the preceding position. Overnight exposure is counted
    at weekday 17:00 rolls, including Friday rolls. No financing is applied.
    """
    def __init__(self, data):
        self.data = data
        index = data.mids.index
        self.index = index
        self.ret = data.returns.fillna(0.).to_numpy()
        self.weights = data.loadings.to_numpy()
        self.spreads = data.spreads.to_numpy()
        local = index.tz_convert('America/New_York')
        date = (local.tz_localize(None)+pd.Timedelta(hours=7)).normalize()
        self.keep = date.dayofweek < 5
        self.codes, self.dates = pd.factorize(date[self.keep], sort=True)
        self.n_dates = len(self.dates)
        self.minute_dates = date
        rolls = pd.date_range(local.min().normalize(), local.max().normalize(), freq='B') + pd.Timedelta(hours=17)
        rolls = rolls[(rolls > local.min()) & (rolls < local.max())]
        locations = local.searchsorted(rolls, side='left')
        self.roll_locations = locations
        self.roll_times = rolls

    def _daily(self, values):
        return np.bincount(self.codes, weights=values[self.keep], minlength=self.n_dates)

    def run(self, signal, delay=0):
        if delay < 0 or int(delay) != delay:
            raise ValueError('Delay must be a non-negative integer number of quote rows.')
        q = signal.reindex(self.index, method='ffill').fillna(0.).to_numpy().copy()
        if delay:
            q[delay:], q[:delay] = q[:-delay].copy(), 0.
        q[-1] = 0.  # Charge final liquidation.
        previous = np.r_[0., q[:-1]]
        changes = q-previous
        traded = np.flatnonzero(changes)
        legs = changes[traded, None]*self.weights
        spread_cost = np.zeros(len(q))
        spread_cost[traded] = (legs*np.log1p(np.sign(legs)*self.spreads[traded]*.5)).sum(axis=1)
        turnover = np.abs(changes)*np.abs(self.weights).sum()
        gross = previous*self.ret
        roll_exposure = np.zeros(len(q))
        held_at_roll = q[self.roll_locations-1]
        np.add.at(roll_exposure, self.roll_locations, np.abs(held_at_roll)*np.abs(self.weights).sum())
        frame = pd.DataFrame({
            'gross': self._daily(gross), 'cost': self._daily(spread_cost),
            'turnover': self._daily(turnover),
            'rollover_exposure': self._daily(roll_exposure),
            'active_minutes': self._daily((q != 0).astype(float)),
        }, index=self.dates)
        frame['net'] = frame.gross-frame.cost
        events = pd.DataFrame({'timestamp': self.index[traded], 'before': previous[traded],
                               'after': q[traded], 'change': changes[traded],
                               'cost': spread_cost[traded]})
        return frame, events


def daily_bootstrap(pnl, draws=2000, block_days=20, seed=1907):
    """Conditional Sharpe intervals with degenerate sparse-trade draws counted.

    A resample containing no PnL has undefined Sharpe. Report how often that
    occurs instead of letting one such draw silently erase every quantile.
    """
    values = pnl.to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), (draws, int(np.ceil(len(values)/block_days))))
    indices = (starts[..., None]+np.arange(block_days)) % len(values)
    samples = values[indices.reshape(draws, -1)[:, :len(values)]]
    std = samples.std(axis=1, ddof=1)
    valid = np.isfinite(std) & (std > 0)
    sr = np.sqrt(252)*samples[valid].mean(axis=1)/std[valid]
    quantiles = np.quantile(sr, [.025, .5, .975]) if len(sr) else [np.nan]*3
    return dict(zip(['low','median','high'], quantiles), draws=draws,
                nonzero_vol_draws=int(valid.sum()), zero_vol_draws=int((~valid).sum()))
