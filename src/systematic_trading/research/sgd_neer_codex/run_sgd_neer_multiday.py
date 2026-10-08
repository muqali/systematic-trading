"""Run fixed day-scale / z-threshold grid using the exported M1 history.

From repo root: poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_multiday.py
Forward-point adjustments are explicitly excluded at the user's request.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.sgd_neer_multiday import (
    DailyPolicy, daily_marks, daily_features, daily_signal, MinuteAccountant, daily_bootstrap,
)
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE as MINUTE_CACHE, daily_metrics, PERIODS

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'sgd_neer_results'/'multiday'
CACHE = HERE.parents[3]/'tmp'/'sgd_neer_multiday'


def policies():
    return [DailyPolicy(f'{kind}_{h}d_z{z:g}', kind=kind, horizon_days=h, entry_z=z,
                        max_hold_days=max(2, 2*h))
            for kind in ['return', 'equilibrium', 'forecast']
            for h in [1, 3, 5, 10] for z in [1.5, 2., 2.5, 3.]]


def summarize(daily, events, name, delay, periods):
    rows = []
    ts = pd.to_datetime(events.timestamp, utc=True).dt.tz_convert('America/New_York')
    for period in periods:
        start, end = PERIODS[period]
        part = daily.loc[start:end]
        e = events.loc[(ts.dt.year >= int(start)) & (ts.dt.year <= int(end))]
        row = dict(policy=name, fill_delay_minutes=delay, period=period, **daily_metrics(part.net))
        row.update(gross_sharpe=daily_metrics(part.gross)['sharpe'],
                   total_gross=float(part.gross.sum()), total_cost=float(part.cost.sum()),
                   trade_days=int((part.turnover > 0).sum()),
                   active_days=int((part.active_minutes > 0).sum()),
                   completed_trades=int(((e.before != 0) & (e.after == 0)).sum()),
                   avg_daily_turnover=float(part.turnover.mean()),
                   days_with_rollover_exposure=int((part.rollover_exposure > 0).sum()),
                   break_even_cost_multiple=part.gross.sum()/part.cost.sum() if part.cost.sum() else np.nan)
        rows.append(row)
    return rows


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    candidates = policies()
    protocol = {
        'policies': [asdict(p) for p in candidates], 'periods': PERIODS,
        'quote_frequency': '1min', 'decision_frequency': 'Daily at 16:00 New York',
        'horizons': 'Fixed business-day grid; weekends skipped, missing/holiday marks retained as NaN.',
        'holding': 'No daily or weekend flattening. Requested maximum hold is 2*horizon business days.',
        'missing_quotes': 'Skip unavailable price-based decisions. Time-limit exits can be requested without a mark; '
                          'execute at next available minute quote, retaining all intervening gap PnL.',
        'risk': '5% annual target using prior daily-return EWM volatility (20-day half life), cap 3 basket units.',
        'entry': 'Absolute z >= entry threshold and absolute edge > 1.5 times basket full spread.',
        'exit': 'Return/equilibrium near mean at 0.25z or crossing; forecast sign reversal or |z|<=0.25; max age.',
        'z_definition': 'Return/equilibrium use empirical past daily signal dispersion (63-day half life). '
                        'Forecast uses trailing 126-day mean/std, minimum 60 predictions, shifted one day.',
        'forecast': 'Weekly regression of H-day future on H-day past return; preceding 252 business dates, '
                    'minimum 100 mature labels, beta clipped <=0, no traded intercept.',
        'financing': 'NO forward-point adjustment, explicitly requested by user.',
        'selection': 'Top three minimum net Sharpe across 2017-2022, 2021-2022, 2023; '
                     '>=10 completed validation trades and >=40 validation active days.',
        'evaluation': '2024-2025 already inspected. All 48 grid results disclosed after selection; '
                      'selection and ensemble do not use final-period results.',
        'warmup': '2016 initial history; forecast z normalization also needs 60 prior predictions.',
        'source': str(MINUTE_CACHE/'data.pkl'),
    }
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    manifest = json.loads((HERE/'sgd_neer_results/minute/data_manifest.json').read_text())
    (OUTPUT/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    source_paths = [Path(__file__), HERE/'sgd_neer_multiday.py']
    signature = hashlib.sha256(b''.join(p.read_bytes() for p in source_paths)).hexdigest()
    valid_cache = (OUTPUT/'run_signature.txt').exists() and (OUTPUT/'run_signature.txt').read_text() == signature
    data = pd.read_pickle(MINUTE_CACHE/'data.pkl')
    assert data.freq == pd.Timedelta(minutes=1)
    accountant = MinuteAccountant(data)
    marks = daily_marks(data)
    marks.to_csv(OUTPUT/'daily_marks.csv')
    coverage = marks.level.notna().groupby(marks.index.year).agg(['count', 'sum', 'mean'])
    coverage.to_csv(OUTPUT/'decision_quote_coverage.csv')
    print('Daily decision quote coverage:', coverage.round(3).to_string(), flush=True)
    features = {}
    signals = {}
    daily_results = {}
    ledger = {}
    development = []
    for policy in candidates:
        key = (policy.kind, policy.horizon_days)
        if key not in features:
            features[key], fits = daily_features(marks, *key)
            features[key].to_csv(OUTPUT/f'features_{key[0]}_{key[1]}d.csv')
            if not fits.empty:
                fits.to_csv(OUTPUT/f'fits_{key[1]}d.csv', index=False)
        signal = daily_signal(marks, features[key], policy)
        signals[policy.name] = signal
        signal.to_pickle(CACHE/f'{policy.name}.pkl')
        for delay in [0, 1]:
            path = OUTPUT/f'daily_{policy.name}_fill{delay}.csv.gz'
            events_path = OUTPUT/f'trades_{policy.name}_fill{delay}.csv.gz'
            if valid_cache and path.exists() and events_path.exists():
                daily = pd.read_csv(path, index_col=0, parse_dates=True)
                events = pd.read_csv(events_path)
            else:
                daily, events = accountant.run(signal, delay)
                daily.to_csv(path, compression='gzip')
                events.to_csv(events_path, index=False, compression='gzip')
            daily_results[(policy.name, delay)] = daily
            ledger[(policy.name, delay)] = events
            development.extend(summarize(daily, events, policy.name, delay,
                                         ['development', 'recent_development', 'validation']))
        if policy.entry_z == 3.:
            print('Completed', policy.kind, policy.horizon_days, 'business-day horizon, all four thresholds.', flush=True)
        pd.DataFrame(development).to_csv(OUTPUT/'comparison_development.csv', index=False)
    dev = pd.DataFrame(development)
    primary = dev[dev.fill_delay_minutes == 0]
    board = primary.pivot(index='policy', columns='period', values='sharpe')
    val = primary.query("period == 'validation'").set_index('policy')
    board['validation_completed_trades'] = val.completed_trades
    board['validation_active_days'] = val.active_days
    board['score'] = board[['development', 'recent_development', 'validation']].min(axis=1, skipna=False)
    board['eligible'] = (board.validation_completed_trades >= 10) & (board.validation_active_days >= 40) & np.isfinite(board.score)
    eligible = board[board.eligible].sort_values('score', ascending=False)
    selected = list(eligible.head(3).index)
    freeze = {'selected': selected,
              'positive_in_all_selection_periods': list(eligible[eligible.score > 0].index),
              'selection': protocol['selection'], 'evaluation': protocol['evaluation']}
    (OUTPUT/'frozen_selection.json').write_text(json.dumps(freeze, indent=2))
    board.sort_values('score', ascending=False).to_csv(OUTPUT/'leaderboard.csv')
    print('Frozen shortlist:', selected, flush=True)
    if selected:
        ensemble = sum(signals[n] for n in selected)/len(selected)
        signals['frozen_ensemble'] = ensemble
        for delay in [0, 1]:
            daily, events = accountant.run(ensemble, delay)
            daily.to_csv(OUTPUT/f'daily_frozen_ensemble_fill{delay}.csv.gz', compression='gzip')
            events.to_csv(OUTPUT/f'trades_frozen_ensemble_fill{delay}.csv.gz', index=False, compression='gzip')
            daily_results[('frozen_ensemble', delay)], ledger[('frozen_ensemble', delay)] = daily, events
    evaluation = []
    for (name, delay), daily in daily_results.items():
        evaluation.extend(summarize(daily, ledger[(name, delay)], name, delay, list(PERIODS)))
    frame = pd.DataFrame(evaluation)
    frame.to_csv(OUTPUT/'comparison_evaluation.csv', index=False)
    # Disclose cost/uncertainty sensitivities for the complete declared grid.
    stress_names = list(signals)
    stress = []
    intervals = []
    for name in stress_names:
        for delay in [0, 1]:
            d = daily_results[(name, delay)].loc['2024':'2025']
            for label, mult, slip in [('base', 1., 0.), ('spread_x1.5', 1.5, 0.),
                                      ('spread_x2', 2., 0.), ('extra_0.25bp_per_leg', 1., .25)]:
                pnl = d.gross-mult*d.cost-d.turnover*slip/1e4
                stress.append(dict(policy=name, fill_delay_minutes=delay, scenario=label, **daily_metrics(pnl)))
            intervals.append(dict(policy=name, fill_delay_minutes=delay, **daily_bootstrap(d.net)))
        if name.endswith('_z2') or '_1d_' in name or name in selected:
            delayed, _ = accountant.run(signals[name], 15)
            delayed.to_csv(OUTPUT/f'daily_{name}_fill15.csv.gz', compression='gzip')
            stress.append(dict(policy=name, fill_delay_minutes=15, scenario='base', **daily_metrics(delayed.net.loc['2024':'2025'])))
    pd.DataFrame(stress).to_csv(OUTPUT/'stress_evaluation.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUTPUT/'bootstrap.csv', index=False)
    # Descriptive day-scale regression diagnostics, with overlapping-label HAC.
    import statsmodels.api as sm
    diagnostics = []
    for h in [1,3,5,10]:
        sample = pd.DataFrame({'x': marks.level.diff(h), 'y': marks.level.shift(-h)-marks.level})
        for period in ['development','validation','evaluation']:
            start, end = PERIODS[period]
            part = sample.loc[start:end].dropna()
            # Restrict forward endpoints to the descriptive fold as well.
            end_year = marks.index.to_series().shift(-h).reindex(part.index).dt.year
            part = part.loc[end_year <= int(end)]
            fit = sm.OLS(part.y, sm.add_constant(part.x)).fit(cov_type='HAC', cov_kwds={'maxlags':2*h})
            diagnostics.append(dict(period=period, horizon_days=h, beta=fit.params.x, hac_t=fit.tvalues.x,
                                    r_squared=fit.rsquared, n=len(part)))
    pd.DataFrame(diagnostics).to_csv(OUTPUT/'return_regression.csv', index=False)
    (OUTPUT/'code_manifest.json').write_text(json.dumps([
        {'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths], indent=2))
    (OUTPUT/'run_signature.txt').write_text(signature)
    result = frame[(frame.period == 'evaluation') & (frame.fill_delay_minutes == 0)]
    print('Evaluation of all predeclared configurations (selection already frozen):', flush=True)
    print(result[['policy','sharpe','gross_sharpe','completed_trades','active_days']].round(3).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
