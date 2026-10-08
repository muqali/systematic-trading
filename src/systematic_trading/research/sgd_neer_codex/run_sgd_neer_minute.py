"""Minute-quote follow-up. Run from repo root with poetry run python <this file>.

Stages are resumable, with inputs and full signals in tmp/sgd_neer_m1_research.
Durable protocols, fits, daily PnL and summaries live in sgd_neer_results/minute.
The final dates have already been inspected in the preceding M15 research.
"""
import argparse
from dataclasses import asdict, replace
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.resampled_data_loader import get_resampled_quotes, RESAMPLED_DATA_DIR
from research.sgd_neer_codex.sgd_neer_backtest import (
    Policy, prepare_basket, causal_features, policy_signal, rolling_forecast,
    backtest, daily_pnl, block_bootstrap_sharpe, rescale_policy,
)
from research.sgd_neer_codex.run_sgd_neer_research import PAIRS, BASELINE_PARAMS
from strategy.sgd_neer_strategy import SGDNEERStrategy

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'sgd_neer_results'/'minute'
CACHE = HERE.parents[3]/'tmp'/'sgd_neer_m1_research'
FREQ = pd.Timedelta(minutes=1)
FROZEN_M15 = ['equilibrium_240m', 'band_720m_cost2', 'band_240m_cost2']
PERIODS = {'development': ('2017', '2022'), 'recent_development': ('2021', '2022'),
           'validation': ('2023', '2023'), 'evaluation': ('2024', '2025'),
           '2024': ('2024', '2024'), '2025': ('2025', '2025'),
           'all_2017_2025': ('2017', '2025')}


def candidates():
    policies = []
    for kind, horizons in [('reversal', [5, 15, 30, 120, 480]),
                           ('equilibrium', [15, 60, 240, 960])]:
        for h in horizons:
            policies.append(Policy(f'{kind}_{h}m', kind=kind, horizon=h, entry_z=2.,
                                   forecast_horizon=240, max_hold=min(2880, max(480, 4*h)), execution_delay=0))
    for h in [30, 60, 240, 720]:
        for c in [1., 2.]:
            policies.append(Policy(f'band_{h}m_cost{c:g}', kind='forecast_band',
                                   horizon=120, max_hold=1440,
                                   forecast_horizon=h, training_weeks=52,
                                   cost_hurdle=c, execution_delay=0))
    return {p.name: p for p in policies}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    protocol = {
        'freq': str(FREQ), 'pairs': PAIRS, 'start': '2016-01-01', 'end': '2025-12-31',
        'policies': [asdict(p) for p in candidates().values()],
        'periods': PERIODS, 'primary_execution_delay_minutes': 0,
        'selection': 'Top three min(development, recent_development, validation) net Sharpe, '
                     'at least 40 validation trade days. Freeze before final scoring.',
        'previously_frozen_M15_benchmarks': FROZEN_M15,
        'evaluation_status': '2024-2025 previously inspected: follow-up, not untouched holdout.',
        'feature_durations_minutes': [60, 240, 960, 2880],
        'volatility': '10 trading days half life and minimum history, 20-day rolling floor; '
                      '1440 bars per 24-hour trading day. Exclude gap jumps.',
        'cache_warning': 'Exporter forward fills up to 10 minutes; no original tick age in cache.',
    }
    write_json(OUTPUT/'protocol.json', protocol)
    if (CACHE/'data.pkl').exists() and (CACHE/'legacy_corrected.pkl').exists():
        print('Using prepared minute data.', flush=True)
        return
    manifest = []
    for p in PAIRS:
        for year in range(2016, 2026):
            path = RESAMPLED_DATA_DIR/f'DAT_ASCII_{p}_M1_{year}.csv'
            manifest.append({'file': str(path), 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    print('Loading all 50 M1 quote caches.', flush=True)
    prices = get_resampled_quotes(PAIRS, dt.date(2016, 1, 1), dt.date(2025, 12, 31), freq=FREQ)
    data = prepare_basket(prices, freq=FREQ)
    pd.to_pickle(data, CACHE/'data.pkl')
    union = pd.DataFrame({p: q.mid for p, q in prices.items()})
    quality = []
    for year, part in union.groupby(union.index.year):
        common = data.mids.loc[str(year)]
        quality.append({'year': year, 'union_rows': len(part), 'common_rows': len(common),
                        'common_fraction': len(common)/len(part),
                        **{f'{p}_missing': int(part[p].isna().sum()) for p in PAIRS}})
    pd.DataFrame(quality).to_csv(OUTPUT/'data_quality.csv', index=False)
    write_json(OUTPUT/'data_manifest.json', {'audit': data.audit, 'files': manifest,
                                          'loadings': data.loadings.to_dict()})
    print('Quote audit:', data.audit, flush=True)
    print('Generating corrected original strategy on the 1-minute quotes.', flush=True)
    baseline = SGDNEERStrategy(PAIRS, prices, BASELINE_PARAMS)
    # Scalar index target avoids storing five copies of a fixed basket loading.
    from research.sgd_neer import build_rets_vs_sgd
    ret = build_rets_vs_sgd(prices)['index'].dropna()
    signal = baseline._index_signal(ret).reindex(data.mids.index).fillna(0.)
    signal.to_pickle(CACHE/'legacy_corrected.pkl')
    del prices, union, baseline
    gc.collect()


def load():
    data = pd.read_pickle(CACHE/'data.pkl')
    if not (CACHE/'features.pkl').exists():
        print('Computing duration-preserving minute features.', flush=True)
        pd.to_pickle(causal_features(data), CACHE/'features.pkl')
    features = pd.read_pickle(CACHE/'features.pkl')
    return data, features


def signal_for(data, features, policy):
    path = CACHE/f'{policy.name}_train_delay{policy.execution_delay}.pkl'
    if path.exists():
        return pd.read_pickle(path)
    prediction = None
    if policy.kind.startswith('forecast'):
        key = f'forecast_{policy.forecast_horizon}_{policy.training_weeks}_d{policy.execution_delay}'
        pred_path = CACHE/f'{key}.pkl'
        if pred_path.exists():
            prediction = pd.read_pickle(pred_path)
        else:
            print('Fitting', key, flush=True)
            prediction, fits = rolling_forecast(
                data, policy.forecast_horizon, policy.training_weeks,
                execution_delay=policy.execution_delay, feature_cache=features)
            prediction.to_pickle(pred_path)
            fits.to_csv(OUTPUT/f'{key}_fits.csv', index=False)
    print('Generating', policy.name, 'training delay', policy.execution_delay, flush=True)
    signal = policy_signal(data, policy, prediction, feature_cache=features)
    signal.to_pickle(path)
    return signal


def daily_backtest(data, signal, name, delay=0, execution_allowed=None):
    path = OUTPUT/f'daily_{name}_fill{delay}.csv.gz'
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    bt = backtest(data, signal, delay=delay, execution_allowed=execution_allowed)
    daily = daily_pnl(bt)
    daily.to_csv(path, compression='gzip')
    return daily


def daily_metrics(d):
    sd = d.std()
    curve = d.cumsum()
    return {'sharpe': float(np.sqrt(252)*d.mean()/sd) if sd > 0 else np.nan,
            'annual_pnl': float(d.mean()*252), 'annual_vol': float(sd*np.sqrt(252)),
            'max_drawdown': float((curve-curve.cummax().clip(lower=0)).min()), 'days': len(d)}


def summarize(daily, name, delay, periods):
    rows = []
    for period in periods:
        start, end = PERIODS[period]
        part = daily.loc[start:end]
        row = dict(policy=name, fill_delay_minutes=delay, period=period, **daily_metrics(part.net))
        row.update(gross_sharpe=daily_metrics(part.gross)['sharpe'],
                   total_gross=float(part.gross.sum()), total_cost=float(part.cost.sum()),
                   avg_daily_turnover=float(part.turnover.mean()),
                   trade_days=int((part.turnover > 0).sum()),
                   days_with_rollover_exposure=int((part.rollover_exposure > 0).sum()),
                   break_even_cost_multiple=float(part.gross.sum()/part.cost.sum()) if part.cost.sum() else np.nan)
        rows.append(row)
    return rows


def development():
    data, features = load()
    rows = []
    periods = ['development', 'recent_development', 'validation']
    for name, policy in [("legacy_corrected", None), *candidates().items()]:
        signal = (pd.read_pickle(CACHE/'legacy_corrected.pkl') if policy is None
                  else signal_for(data, features, policy))
        for delay in [0, 1]:
            daily = daily_backtest(data, signal, name, delay)
            rows.extend(summarize(daily, name, delay, periods))
        pd.DataFrame(rows).to_csv(OUTPUT/'comparison_development.csv', index=False)
        print(name, 'development complete.', flush=True)
    frame = pd.DataFrame(rows)
    primary = frame[(frame.fill_delay_minutes == 0) & (frame.policy != 'legacy_corrected')]
    board = primary.pivot(index='policy', columns='period', values='sharpe')
    board['score'] = board[periods].min(axis=1, skipna=False)
    board['validation_trade_days'] = primary.query("period == 'validation'").set_index('policy').trade_days
    eligible = board[(board.validation_trade_days >= 40) & np.isfinite(board.score)].sort_values('score', ascending=False)
    selected = list(eligible.head(3).index)
    if not selected:
        raise ValueError('No adequately active policies to evaluate.')
    write_json(OUTPUT/'frozen_selection.json', {
        'selected': selected, 'positive_in_all_selection_periods': list(eligible[eligible.score > 0].index),
        'selection': 'Top three minimum net Sharpe across development, recent development and validation; '
                     'at least 40 validation trade days.',
        'evaluation_status': 'Follow-up on previously inspected 2024-2025 dates.',
    })
    board.sort_values('score', ascending=False).to_csv(OUTPUT/'leaderboard.csv')
    print('FROZEN SHORTLIST:', selected, flush=True)
    print(board.sort_values('score', ascending=False).round(3).to_string(), flush=True)


def changed_quote_gate(data, max_age='2min'):
    """Causal freshness proxy, NOT proof of actual tick age/executability."""
    changed = data.mids.ne(data.mids.shift()) | data.spreads.ne(data.spreads.shift())
    timestamp = pd.Series(data.mids.index, index=data.mids.index)
    return pd.DataFrame({p: (timestamp-timestamp.where(changed[p]).ffill()) <= pd.Timedelta(max_age)
                         for p in data.mids}).all(axis=1)


def gated_targets(data, signal, allowed):
    # Defer the whole basket change until every leg satisfies the proxy.
    # Existing exposure stays live and earns all gap/unchanged-quote returns.
    signal = signal.where(allowed).ffill().fillna(0.)
    return signal


def evaluation():
    freeze = json.loads((OUTPUT/'frozen_selection.json').read_text())
    selected = freeze['selected']
    names = list(dict.fromkeys(['legacy_corrected', *FROZEN_M15, *selected]))
    data, features = load()
    fresh = changed_quote_gate(data)
    policies = candidates()
    signals = {name: (pd.read_pickle(CACHE/'legacy_corrected.pkl') if name == 'legacy_corrected'
                     else signal_for(data, features, policies[name])) for name in names}
    signals['frozen_ensemble'] = sum(signals[n] for n in selected)/len(selected)
    rows, stress = [], []
    for name, signal in signals.items():
        for delay in [0, 1, 2, 15]:
            daily = daily_backtest(data, signal, name, delay)
            rows.extend(summarize(daily, name, delay, list(PERIODS)))
            if delay in [0, 1]:
                part = daily.loc['2024':'2025']
                for scenario, cost, slip, roll in [('base', 1., 0., 0.), ('spread_x1.5', 1.5, 0., 0.),
                                                   ('spread_x2', 2., 0., 0.),
                                                   ('extra_0.25bp_per_leg', 1., .25, 0.),
                                                   ('1bp_per_roll_exposure', 1., 0., 1.)]:
                    net = part.gross-part.cost*cost-part.turnover*slip/1e4-part.rollover_exposure*roll/1e4
                    stress.append(dict(policy=name, fill_delay_minutes=delay, scenario=scenario, **daily_metrics(net)))
        for scenario, target, allowed in [('fresh_at_fill_2min', signal, fresh),
                                          ('decisions_every_15min', gated_targets(data, signal, data.mids.index.minute % 15 == 0), None)]:
            for delay in [0, 1]:
                daily = daily_backtest(data, target, name+'_'+scenario, delay, execution_allowed=allowed)
                stress.append(dict(policy=name, fill_delay_minutes=delay, scenario=scenario,
                                   **daily_metrics(daily.net.loc['2024':'2025'])))
        pd.DataFrame(rows).to_csv(OUTPUT/'comparison_evaluation.csv', index=False)
        pd.DataFrame(stress).to_csv(OUTPUT/'stress_evaluation.csv', index=False)
        print(name, 'evaluation and stress complete.', flush=True)
    # Refit the selected forecast labels to the actual one-minute delay as well.
    delayed = []
    for name in selected:
        p = replace(policies[name], execution_delay=1)
        sig = signal_for(data, features, p)
        daily = daily_backtest(data, sig, name+'_refit_delay1', 1)
        delayed.extend(summarize(daily, name, 1, list(PERIODS)))
    pd.DataFrame(delayed).to_csv(OUTPUT/'refitted_delay1.csv', index=False)
    # Bootstrap daily PnL without re-annualising the observation count at M1.
    intervals = []
    for name in signals:
        for delay in [0, 1]:
            d = pd.read_csv(OUTPUT/f'daily_{name}_fill{delay}.csv.gz', index_col=0, parse_dates=True).net.loc['2024':'2025']
            # block_bootstrap_sharpe expects intraday timezone-aware observations.
            d.index = d.index.tz_localize('America/New_York')
            intervals.append(dict(policy=name, fill_delay_minutes=delay, **block_bootstrap_sharpe(d)))
    pd.DataFrame(intervals).to_csv(OUTPUT/'bootstrap.csv', index=False)
    frame = pd.DataFrame(rows)
    print(frame.query("period == 'evaluation'").pivot(index='policy', columns='fill_delay_minutes', values='sharpe').round(3).to_string(), flush=True)


def diagnostics():
    data, features = load()
    group = features[3]
    rows = []
    for period in ['development', 'validation', 'evaluation']:
        start, end = PERIODS[period]
        base_var = data.returns.where(data.continuous).loc[start:end].var()
        for h in [1, 2, 5, 15, 30, 60, 120, 240, 480, 1200]:
            ret = data.level.diff(h).where(group.eq(group.shift(h))).loc[start:end]
            rows.append(dict(period=period, minutes=h, variance_ratio=ret.var()/(h*base_var), observations=ret.count()))
    pd.DataFrame(rows).to_csv(OUTPUT/'variance_ratio.csv', index=False)
    # Sample regression anchors every 30 minutes, retaining minute-resolution
    # forward returns; HAC then needs far fewer lags than 3.6m overlapping labels.
    import statsmodels.api as sm
    rows = []
    anchor = data.level.index.minute % 30 == 0
    for past in [15, 120, 480]:
        x = data.level.diff(past).where(group.eq(group.shift(past)))
        for future in [5, 30, 120]:
            for delay in [0, 1, 15]:
                y = (data.level.shift(-future-delay)-data.level.shift(-delay)).where(group.eq(group.shift(-future-delay)))
                sample = pd.concat({'x': x, 'y': y}, axis=1).loc[anchor].dropna()
                for period in ['development', 'validation', 'evaluation']:
                    start, end = PERIODS[period]
                    part = sample.loc[start:end]
                    fit = sm.OLS(part.y, sm.add_constant(part.x)).fit(cov_type='HAC', cov_kwds={'maxlags': int(np.ceil((past+future+delay)/30))})
                    rows.append(dict(period=period, past_minutes=past, future_minutes=future, delay_minutes=delay,
                                     beta=fit.params.x, hac_t=fit.tvalues.x, r_squared=fit.rsquared, n=len(part)))
    pd.DataFrame(rows).to_csv(OUTPUT/'return_regression.csv', index=False)
    fresh = changed_quote_gate(data)
    pd.DataFrame({'fresh_proxy': fresh, 'features_usable': features[0].notna().all(axis=1)}).groupby(data.mids.index.year).mean().to_csv(OUTPUT/'feature_coverage.csv')
    print('Diagnostics complete.', flush=True)


def controls():
    """15-minute models on the SAME source, executed on the minute quote grid.

    This separates a shorter execution delay from updating/refitting the model
    on every minute. Parameters are the pre-existing frozen M15 benchmarks.
    """
    data = pd.read_pickle(CACHE/'data.pkl')
    grid = pd.date_range(data.mids.index.min().floor('15min'), data.mids.index.max(), freq='15min')
    grid = grid[(grid.dayofweek < 4) | ((grid.dayofweek == 4) & (grid.hour < 17))
                | ((grid.dayofweek == 6) & (grid.hour >= 18))]
    prices = {}
    for p in PAIRS:
        # Keep absent grid marks as NaN before the original strategy computes
        # returns, matching its M1 preprocessing. Research models subsequently
        # intersect valid marks and explicitly retain held-position gap PnL.
        mid = data.mids[p].reindex(grid)
        half = data.spreads[p].reindex(grid)/2
        prices[p] = pd.DataFrame({'mid': mid, 'bid': mid*(1-half), 'ask': mid*(1+half)})
    coarse = prepare_basket(prices, freq='15min')
    features = causal_features(coarse)
    from research.sgd_neer import build_rets_vs_sgd
    original = SGDNEERStrategy(PAIRS, prices, BASELINE_PARAMS)
    signals = {'legacy_corrected': original._index_signal(build_rets_vs_sgd(prices)['index'].dropna())}
    for name in FROZEN_M15:
        policy = rescale_policy(candidates()[name], freq='15min', reference_freq='1min')
        prediction = None
        if policy.kind.startswith('forecast'):
            prediction, fits = rolling_forecast(coarse, policy.forecast_horizon, policy.training_weeks,
                                                execution_delay=0, feature_cache=features)
            fits.to_csv(OUTPUT/f'control_{name}_fits.csv', index=False)
        signals[name] = policy_signal(coarse, policy, prediction, feature_cache=features)
    rows = []
    for name, signal in signals.items():
        target = signal.reindex(data.mids.index, method='ffill').fillna(0.)
        for delay in [0, 1, 15]:
            daily = daily_backtest(data, target, 'control15m_grid_'+name, delay)
            rows.extend(summarize(daily, name, delay, list(PERIODS)))
        pd.DataFrame(rows).to_csv(OUTPUT/'comparison_15min_controls.csv', index=False)
        print('Matched 15-minute control:', name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'prepare', 'development', 'evaluation', 'diagnostics', 'controls'], default='all')
    args = parser.parse_args()
    if args.stage in ['all', 'prepare']:
        prepare()
    if args.stage in ['all', 'development']:
        development()
    if args.stage in ['all', 'evaluation']:
        evaluation()
    if args.stage in ['all', 'diagnostics']:
        diagnostics()
    if args.stage in ['all', 'controls']:
        controls()
