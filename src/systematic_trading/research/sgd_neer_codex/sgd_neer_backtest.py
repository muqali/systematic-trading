"""Causal, spread-aware research on a tradable SGD basket proxy.

Signals observe a completed bar and fill at its quote (delay=0) or a later quote.
Positions always earn only the *subsequent* return. PnL uses the repository's log-return
notional convention, not a cash/spot settlement ledger. No financing is assumed:
the research policies request liquidation before the New York daily rollover.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from research.sgd_neer import normalised_weights


def basket_loadings(pairs, weights=None):
    """Pair notionals replicating one unit of the normalised SGD-base index."""
    weights = normalised_weights(list(pairs), weights)
    if "USDSGD" not in pairs:
        raise ValueError("USDSGD is required.")
    return pd.Series({
        p: -1.0 if p == "USDSGD" else w if p.startswith("USD") else -w
        for p, w in weights.items()
    })


@dataclass
class BasketData:
    mids: pd.DataFrame
    spreads: pd.DataFrame
    loadings: pd.Series
    level: pd.Series
    returns: pd.Series
    full_spread: pd.Series
    continuous: pd.Series
    audit: dict
    freq: pd.Timedelta = pd.Timedelta('15min')

    @property
    def bars_per_day(self):
        return int(pd.Timedelta(days=1) / self.freq)

    def bars(self, duration):
        ratio = pd.Timedelta(duration) / self.freq
        if ratio < 1 or not float(ratio).is_integer():
            raise ValueError('Duration must be a positive integer number of quote bars.')
        return int(ratio)


def prepare_basket(prices, freq="15min"):
    """Intersect valid quotes, preserving the entire return across quote gaps.

    No forward/back filling: missing marks cannot manufacture free PnL. Features
    reset after gaps; held positions still earn gap returns at the next mark.
    """
    freq = pd.Timedelta(freq)
    if freq <= pd.Timedelta(0) or pd.Timedelta(days=1) % freq:
        raise ValueError('freq must be positive and divide a day exactly.')
    pairs = list(prices)
    for p, q in prices.items():
        if not isinstance(q.index, pd.DatetimeIndex) or q.index.tz is None:
            raise ValueError(f"{p}: timezone-aware DatetimeIndex required.")
        if q.index.has_duplicates or not q.index.is_monotonic_increasing:
            raise ValueError(f"{p}: sorted, unique timestamps required.")
    mids = pd.DataFrame({p: q["mid"] for p, q in prices.items()})
    bids = pd.DataFrame({p: q["bid"] for p, q in prices.items()})
    asks = pd.DataFrame({p: q["ask"] for p, q in prices.items()})
    valid = ((bids > 0) & (asks >= bids) & (mids >= bids) & (mids <= asks)).all(axis=1)
    valid &= np.isfinite(mids).all(axis=1) & np.isfinite(asks).all(axis=1)
    audit = {"union_rows": len(mids), "valid_rows": int(valid.sum()),
             "missing_or_invalid_rows": int((~valid).sum())}
    mids, bids, asks = (x.loc[valid] for x in (mids, bids, asks))
    if len(mids) < 2:
        raise ValueError("At least two common valid quotes required.")
    loadings = basket_loadings(pairs)
    level = np.log(mids).mul(loadings).sum(axis=1).rename("level")
    spreads = (asks - bids) / mids
    full_spread = spreads.mul(loadings.abs()).sum(axis=1).rename("full_spread")
    gaps = mids.index.to_series().diff()
    audit["max_gap_hours"] = float(gaps.max() / pd.Timedelta(hours=1))
    audit["gaps_over_one_hour"] = int((gaps > pd.Timedelta(hours=1)).sum())
    audit["median_round_trip_bps"] = float(full_spread.median() * 1e4)
    return BasketData(mids, spreads, loadings, level, level.diff().rename("ret"),
                      full_spread, gaps.eq(freq), audit, freq)


def daily_pnl(pnl):
    """17:00 New York trade date, weekdays, including flat observed weekdays."""
    local = pnl.index.tz_convert("America/New_York").tz_localize(None)
    dates = (local + pd.Timedelta(hours=7)).normalize()
    out = pnl.groupby(dates).sum()
    # Exclude absent dates (notably the 2023 data outage), retain observed flats.
    return out[out.index.dayofweek < 5]


def metrics(pnl):
    d = daily_pnl(pnl)
    std = d.std()
    curve = d.cumsum()
    drawdown = curve - curve.cummax().clip(lower=0)
    return {"sharpe": float(np.sqrt(252) * d.mean() / std) if std > 0 else np.nan,
            "annual_pnl": float(d.mean() * 252),
            "annual_vol": float(std * np.sqrt(252)),
            "max_drawdown": float(drawdown.min()), "days": len(d)}


def signals_to_pairs(data, signals):
    """Align scalar basket units or explicit pair notionals on the quote grid."""
    if isinstance(signals, pd.Series):
        signals = pd.DataFrame({p: signals * w for p, w in data.loadings.items()})
    target = signals.reindex(index=data.mids.index, columns=data.mids.columns).fillna(0.0)
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Signals must be finite.")
    return target


def backtest(data, signals, delay=1, cost_multiple=1.0, extra_slippage_bps=0.0,
             liquidate=True, execution_allowed=None):
    """Cost every trade (including entry and final exit), with explicit fill delay.

    With delay=0, a signal calculated at t trades at t's bid/ask and earns the
    return from t to t+1, never the return used to calculate that signal.
    Additional slippage is charged per absolute traded pair notional.
    No missing price return is dropped from a held position's PnL.
    execution_allowed gates fills after the delay; existing exposure stays live
    until another allowed quote. Final liquidation remains unconditional.
    """
    if delay < 0 or int(delay) != delay:
        raise ValueError("delay must be a non-negative integer.")
    if cost_multiple < 0 or extra_slippage_bps < 0:
        raise ValueError("Costs must be non-negative.")
    target = signals_to_pairs(data, signals)
    position = target.shift(int(delay), fill_value=0.0)
    if execution_allowed is not None:
        allowed = execution_allowed.reindex(position.index).fillna(False).astype(bool)
        position = position.where(allowed, axis=0).ffill().fillna(0.0)
    if liquidate:
        position.iloc[-1] = 0.0
    changes = position.diff().fillna(position)
    pair_ret = np.log(data.mids).diff().fillna(0.0)
    gross = (position.shift(1, fill_value=0.0) * pair_ret).sum(axis=1)
    # Exact directional half-spread log cost, consistent with AggressiveTrader.
    cost = (changes * np.log1p(np.sign(changes) * data.spreads * 0.5)).sum(axis=1)
    turnover = changes.abs().sum(axis=1)
    cost = cost * cost_multiple + turnover * extra_slippage_bps / 1e4
    # Exposure still held across 17:00 NY (including missing-quote exits). This
    # makes the omitted financing observable instead of assuming exits occurred.
    local = data.mids.index.tz_convert("America/New_York")
    rolls = pd.date_range(local.min().normalize(), local.max().normalize(), freq="B")
    rolls = rolls + pd.Timedelta(hours=17)
    locations = local.searchsorted(rolls, side="left")
    rollover_exposure = np.zeros(len(local))
    for loc in locations[(locations > 0) & (locations < len(local))]:
        rollover_exposure[loc] += float(position.iloc[loc-1].abs().sum())
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross-cost,
                         "turnover": turnover,
                         "rollover_exposure": rollover_exposure,
                         "gross_exposure": position.abs().sum(axis=1)})


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str = "reversal"
    horizon: int = 8
    entry_z: float = 2.0
    exit_z: float = 0.25
    cost_hurdle: float = 1.5
    forecast_horizon: int = 16
    training_weeks: int = 26
    max_hold: int = 96
    vol_target: float = 0.05
    max_units: float = 3.0
    session: str = "all"
    execution_delay: int = 1


def rescale_policy(policy, freq, reference_freq='15min'):
    """Preserve economic horizons; execution_delay remains quote bars."""
    ratio = pd.Timedelta(reference_freq) / pd.Timedelta(freq)
    changes = {}
    for field in ['horizon', 'forecast_horizon', 'max_hold']:
        bars = getattr(policy, field)*ratio
        if bars < 1 or not float(bars).is_integer():
            raise ValueError(f'{field} is not representable on the requested quote grid.')
        changes[field] = int(bars)
    return replace(policy, **changes)


def research_candidates():
    """Small, declared design grid; never expanded after the final evaluation."""
    policies = []
    for kind, horizons in [("reversal", [8, 32, 96]), ("equilibrium", [16, 64, 256])]:
        for h in horizons:
            for z in [1.5, 2.0]:
                policies.append(Policy(f"{kind}_{h}_{z:g}", kind=kind, horizon=h,
                                       entry_z=z, max_hold=min(192, max(32, 4*h))))
    for h in [4, 16, 48]:
        for weeks in [12, 52]:
            policies.append(Policy(f"forecast_{h}_{weeks}", kind="forecast",
                                   forecast_horizon=h, training_weeks=weeks,
                                   max_hold=2*h))
    return policies


def causal_features(data):
    # Exclude jumps across missing quotes; retain durations when frequency changes.
    day = data.bars_per_day
    ret = data.returns.where(data.continuous)
    vol = ret.ewm(halflife=day*10, min_periods=day*10).std().shift(1)
    # A floor estimated using past data only, for stable risk sizes in quiet hours.
    vol = vol.clip(lower=vol.rolling(day*20, min_periods=day).median()*0.5)
    group = (~data.continuous).cumsum()
    run = data.continuous.groupby(group).cumsum()
    features = {}
    for reference_bars in [4, 16, 64, 192]:
        h = data.bars(pd.Timedelta(minutes=15*reference_bars))
        # Stable feature names refer to the original 15-minute specification.
        features[f"ret_{reference_bars}"] = (data.level.diff(h)/(vol*np.sqrt(h))).where(run >= h)
    return pd.DataFrame(features).clip(-6, 6), vol, run, group


def rolling_forecast(data, horizon=16, training_weeks=26, ridge=0.1, target_level=None,
                     execution_delay=1, feature_cache=None):
    """Weekly ridge fits using *realised* labels and fold-local transformations.

    Label at t is level[t+d+h]-level[t+d], where d is execution_delay.
    In particular, d=0 includes the immediate next return in the forecast label.
    Training requires its actual end timestamp to precede the fit.
    Gaps invalidate features/labels; there is no full-sample normalisation.
    """
    if execution_delay < 0 or int(execution_delay) != execution_delay:
        raise ValueError("execution_delay must be a non-negative integer.")
    delay = int(execution_delay)
    features, vol, _, group = causal_features(data) if feature_cache is None else feature_cache
    target_level = data.level if target_level is None else target_level.reindex(data.level.index)
    label = (target_level.shift(-delay-horizon)-target_level.shift(-delay)) / (vol*np.sqrt(horizon))
    label = label.where(group.shift(-delay-horizon).eq(group))
    label_end = data.level.index.to_series().shift(-delay-horizon)
    sample = features.assign(y=label.clip(-6, 6)).dropna()
    sample_end = pd.DatetimeIndex(label_end.loc[sample.index])
    sample_x = sample[features.columns].to_numpy()
    sample_y = sample['y'].to_numpy()
    prediction = pd.Series(np.nan, index=data.level.index, name="forecast")
    fits = []
    dates = pd.date_range(data.level.index.min().normalize(),
                          data.level.index.max()+pd.Timedelta(days=8), freq="W-SAT")
    for start, end in zip(dates[:-1], dates[1:]):
        if start < data.level.index.min()+pd.Timedelta(weeks=training_weeks):
            continue
        left = sample.index.searchsorted(start-pd.Timedelta(weeks=training_weeks))
        right = sample_end.searchsorted(start, side='left')
        n = right-left
        if n < data.bars_per_day*20:
            continue
        x, y = sample_x[left:right], sample_y[left:right]
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.maximum(scale, 0.1)
        x = (x-center)/scale
        # No directional intercept: the experiment is about reversal, not carry.
        beta = np.linalg.solve(x.T@x + ridge*len(x)*np.eye(x.shape[1]), x.T@y)
        test = features.iloc[features.index.searchsorted(start):features.index.searchsorted(end)]
        prediction.loc[test.index] = ((test-center)/scale).to_numpy()@beta
        fits.append({"fit_ts": start, "last_label_end": sample_end[right-1],
                     "n": n, "execution_delay": delay,
                     **dict(zip(features.columns, beta))})
    prediction *= vol*np.sqrt(horizon)
    return prediction, pd.DataFrame(fits)


def policy_signal(data, policy, prediction=None, feature_cache=None):
    """Hysteretic positions: enter on sufficient edge, hold size until exit.

    A spread gate controls entry; exits have independent mean-crossing/time
    rules. Daily flattening avoids intentionally assuming free overnight carry.
    """
    _, vol, run, group = causal_features(data) if feature_cache is None else feature_cache
    if policy.kind == "reversal":
        displacement = data.level.diff(policy.horizon).where(run >= policy.horizon)
        z = displacement/(vol*np.sqrt(policy.horizon))
        edge = -0.5*displacement  # fixed heuristic, not a fitted expected return
    elif policy.kind == "equilibrium":
        mean = data.level.groupby(group).transform(
            lambda s: s.ewm(halflife=policy.horizon, adjust=False).mean())
        displacement = (data.level-mean).where(run >= min(policy.horizon, data.bars_per_day))
        z = displacement/(vol*np.sqrt(policy.horizon/(2*np.log(2))))
        edge = -0.5*displacement
    elif policy.kind in ("forecast", "forecast_band"):
        if prediction is None:
            prediction, _ = rolling_forecast(data, policy.forecast_horizon, policy.training_weeks,
                                            execution_delay=policy.execution_delay)
        edge = prediction
        z = -edge/(vol*np.sqrt(policy.forecast_horizon))
    else:
        raise ValueError(f"Unknown policy kind {policy.kind!r}")
    # At 16:30 NY request an exit; with normal 15-minute sampling fill at 16:45.
    # Remain flat until 18:00, avoiding the rollover's exceptional spreads.
    ny = data.mids.index.tz_convert("America/New_York")
    minutes = ny.hour*60+ny.minute
    allowed = (minutes < 16*60+30) | (minutes >= 18*60)
    if policy.session == "asia":
        sg = data.mids.index.tz_convert("Asia/Singapore")
        allowed &= (sg.hour >= 8) & (sg.hour < 17)
    elif policy.session != "all":
        raise ValueError("session must be all or asia.")
    risk_units = (policy.vol_target/(vol*np.sqrt(252*data.bars_per_day))).clip(upper=policy.max_units)
    edge_np, z_np, risk, vol_np = (s.to_numpy() for s in (edge, z, risk_units, vol))
    spread = data.full_spread.to_numpy()
    continuous = data.continuous.to_numpy()
    signal = np.zeros(len(data.mids))
    q, age = 0.0, 0
    for i in range(len(signal)):
        usable = (allowed[i] and continuous[i] and np.isfinite(edge_np[i])
                  and np.isfinite(z_np[i]) and np.isfinite(risk[i]))
        if not usable:
            q, age = 0.0, 0
        elif policy.kind == "forecast_band":
            # Maximise q*mu - risk*q^2/2 - half_spread*|q-q_previous|.
            # Move to the nearest boundary, not all the way to the target.
            curvature = 0.2*vol_np[i]*np.sqrt(policy.forecast_horizon)/risk[i]
            half_cost = 0.5*policy.cost_hurdle*spread[i]
            lower = max(-policy.max_units, min(policy.max_units, (edge_np[i]-half_cost)/curvature))
            upper = max(-policy.max_units, min(policy.max_units, (edge_np[i]+half_cost)/curvature))
            q = max(lower, min(upper, q))
        elif q:
            age += 1
            exit_signal = (np.sign(q)*edge_np[i] <= 0 if policy.kind == "forecast"
                           else np.sign(q)*z_np[i] >= -policy.exit_z)
            if exit_signal or age >= policy.max_hold:
                q, age = 0.0, 0
        else:
            enough_z = policy.kind == "forecast" or abs(z_np[i]) >= policy.entry_z
            if enough_z and abs(edge_np[i]) > policy.cost_hurdle*spread[i]:
                q, age = np.sign(edge_np[i])*risk[i], 0
        signal[i] = q
    return pd.Series(signal, index=data.mids.index, name=policy.name)


def component_forecast_signals(data, policy, pairs=None):
    """Predict executable legs from the basket features; budget risk equally.

    This is intentionally a different strategy from exact basket replication:
    leg forecasts can imply residual currency/directional exposures.
    """
    from dataclasses import replace

    pairs = list(data.mids) if pairs is None else list(pairs)
    if not pairs or not set(pairs).issubset(data.mids.columns):
        raise ValueError("At least one available execution pair is required.")
    signals = pd.DataFrame(0.0, index=data.mids.index, columns=data.mids.columns)
    fits = []
    for pair in pairs:
        level = np.log(data.mids[pair])
        forecast, fit = rolling_forecast(data, policy.forecast_horizon,
                                        policy.training_weeks, target_level=level,
                                        execution_delay=policy.execution_delay)
        leg = replace(data, level=level, returns=level.diff(), full_spread=data.spreads[pair])
        signals[pair] = policy_signal(leg, policy, forecast)/len(pairs)
        fits.append(fit.assign(pair=pair))
    return signals, pd.concat(fits, ignore_index=True)


def mean_reversion_diagnostics(data):
    """Descriptive variance ratios and lag-aware OLS with HAC standard errors."""
    import statsmodels.api as sm

    group = (~data.continuous).cumsum()
    rows = []
    base_var = data.returns.where(data.continuous).var()
    for h in [1, 2, 4, 8, 16, 32, 64, 80]:
        ret = data.level.diff(h).where(group.eq(group.shift(h)))
        rows.append({"bars": h, "minutes": h*data.freq/pd.Timedelta(minutes=1), "variance_ratio": ret.var()/(h*base_var),
                     "observations": int(ret.count())})
    regressions = []
    for past in [4, 8, 32]:
        x = data.level.diff(past).where(group.eq(group.shift(past)))
        for future in [2, 8, 32]:
            for delay in [0, 1, 2]:
                y = (data.level.shift(-delay-future)-data.level.shift(-delay)).where(
                    group.eq(group.shift(-delay-future)))
                sample = pd.concat({"x": x, "y": y}, axis=1).dropna()
                fit = sm.OLS(sample.y, sm.add_constant(sample.x)).fit(
                    cov_type="HAC", cov_kwds={"maxlags": past+future+delay})
                regressions.append({"past_bars": past, "future_bars": future, "delay_bars": delay,
                                    "beta": fit.params.x, "hac_t": fit.tvalues.x,
                                    "r_squared": fit.rsquared, "n": len(sample)})
    return pd.DataFrame(rows), pd.DataFrame(regressions)


def block_bootstrap_sharpe(pnl, draws=2000, block_days=20, seed=1907):
    """Circular block bootstrap; serial dependence is retained within blocks."""
    values = daily_pnl(pnl).to_numpy()
    if len(values) < 2:
        return {"low": np.nan, "median": np.nan, "high": np.nan}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), (draws, int(np.ceil(len(values)/block_days))))
    indices = (starts[..., None]+np.arange(block_days)) % len(values)
    samples = values[indices.reshape(draws, -1)[:, :len(values)]]
    sr = np.sqrt(252)*samples.mean(axis=1)/samples.std(axis=1, ddof=1)
    return dict(zip(["low", "median", "high"], np.quantile(sr, [.025, .5, .975])))
