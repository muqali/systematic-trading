"""Test alternate daily clocks and rolling signals using the M1 basket cache.

Run from repo root with:
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_timing.py
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.sgd_neer_multiday import DailyPolicy, daily_features, daily_signal, MinuteAccountant, daily_bootstrap
from research.sgd_neer_codex.sgd_neer_timing import clock_marks, rolling_marks, rolling_features, RollingPolicy, rolling_signal
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE as MINUTE_CACHE, PERIODS, daily_metrics
from research.sgd_neer_codex.run_sgd_neer_multiday import summarize

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'sgd_neer_results'/'timing'
CACHE = HERE.parents[3]/'tmp'/'sgd_neer_timing'


def declared_policies():
    fixed = [(hour, DailyPolicy(f'clock{hour:02d}_{kind}_{h}d_z{z:g}', kind, h,
                                entry_z=z, max_hold_days=2*h))
             for hour in [0, 4, 8, 12, 16] for kind in ['return', 'equilibrium']
             for h in [1, 3] for z in [1.5, 2.]]
    rolling = [RollingPolicy(f'rolling_{kind}_{h}h_z{z:g}_{mode}', kind, h,
                              entry_z=z, review_hour=hour)
               for kind in ['return', 'equilibrium'] for h in [6, 24, 72, 120, 240]
               for z in [1.5, 2.] for mode, hour in [('continuous', None), ('clock16', 16)]]
    return fixed, rolling


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    fixed, rolling = declared_policies()
    protocol = {
        'fixed_policies': [dict(hour_ny=h, **asdict(p)) for h, p in fixed],
        'rolling_policies': [asdict(p) for p in rolling],
        'periods': PERIODS, 'quote_frequency': '1 minute',
        'daily_clocks': '00:00, 04:00, 08:00, 12:00, 16:00 America/New_York, weekdays.',
        'daily_method': 'Same marks, features, risk sizing and entry/exit rules as prior multi-day study; only clock changes.',
        'rolling_horizons': '6, 24, 72, 120, 240 scheduled market hours. Market grid uses Sunday 18:00 '
                            'through Friday 17:00 New York. Weekend closure is excluded; missing quote minutes remain NaN.',
        'rolling_return': 'level(t)-level(t-H), minus prior EWM mean (63-market-day half life); '
                          'divide by prior empirical EWM standard deviation of H-hour returns.',
        'rolling_equilibrium': 'Deviation from prior EWM log level with H-hour half life, divided by prior '
                               'empirical deviation standard deviation with 63-market-day half life.',
        'rolling_risk': '5% annual target / prior empirical std of 24-market-hour returns / sqrt(252); '
                        '20-market-day half life, minimum 40 market days of available observations, cap 3 basket units.',
        'statistics': 'Rolling estimates update each market minute. All centers/scales lagged one minute; '
                      'overlapping long returns are not treated as independent statistical observations.',
        'rolling_controls': 'Identical rolling features and sizing also reviewed only at 16:00 NY. '
                            'This isolates decision frequency; do not attribute all differences from the old daily '
                            'signal to entry timing, because normalization and horizon definitions also change.',
        'entry_exit': '|z| >= 1.5 or 2, heuristic reversion edge = half displacement > 1.5 full basket spreads; '
                      'exit near/crossing mean (0.25z), or max hold 2H market hours. Constant size per trade, '
                      'no same-minute re-entry; overnight and weekend holding allowed. No overnight flatten.',
        'outages': 'No price interpolation. Skip price decisions on missing marks. Timeouts may request exit '
                   'without quotes; retain intervening gap PnL and execute next available quote.',
        'execution': 'Same-quote primary; one subsequent common quote sensitivity. Delay is quote rows, '
                     'so may exceed one elapsed minute across missing data. Full bid/ask costs, final liquidation.',
        'financing': 'Forward points explicitly excluded at user request.',
        'selection': 'Rank minimum Sharpe across 2017-2022, 2021-2022, 2023. Shortlist at most three with '
                     'positive minimum, >=10 completed trades and >=40 active days in 2023. No optimization on 2024-2025.',
        'evaluation': '2024-2025 previously inspected; report all 80 declared configurations. Exploratory, not a fresh holdout.',
        'source': str(MINUTE_CACHE/'data.pkl'),
    }
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (OUTPUT/'data_manifest.json').write_bytes((HERE/'sgd_neer_results/minute/data_manifest.json').read_bytes())
    paths = [Path(__file__), HERE/'sgd_neer_timing.py', HERE/'sgd_neer_multiday.py',
             HERE/'run_sgd_neer_multiday.py', HERE/'run_sgd_neer_minute.py', HERE/'sgd_neer_backtest.py']
    signature = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    valid_cache = (OUTPUT/'run_signature.txt').exists() and (OUTPUT/'run_signature.txt').read_text() == signature
    data = pd.read_pickle(MINUTE_CACHE/'data.pkl')
    assert data.freq == pd.Timedelta(minutes=1)
    accountant = MinuteAccountant(data)
    rows, metadata, coverage, stress, intervals, entry_times = [], [], [], [], [], []

    def evaluate(signal, info):
        name = signal.name
        metadata.append(dict(policy=name, **info))
        signal.to_pickle(CACHE/f'{name}.pkl')
        for delay in [0, 1]:
            path = OUTPUT/f'daily_{name}_fill{delay}.csv.gz'
            event_path = OUTPUT/f'trades_{name}_fill{delay}.csv.gz'
            if valid_cache and path.exists() and event_path.exists():
                daily = pd.read_csv(path, index_col=0, parse_dates=True)
                events = pd.read_csv(event_path)
            else:
                daily, events = accountant.run(signal, delay)
                daily.to_csv(path, compression='gzip')
                events.to_csv(event_path, index=False, compression='gzip')
            rows.extend(summarize(daily, events, name, delay, list(PERIODS)))
            final = daily.loc['2024':'2025']
            for label, mult, slip in [('base', 1., 0.), ('spread_x2', 2., 0.),
                                      ('extra_0.25bp_per_leg', 1., .25)]:
                pnl = final.gross-mult*final.cost-final.turnover*slip/1e4
                stress.append(dict(policy=name, fill_delay_minutes=delay, scenario=label, **daily_metrics(pnl)))
            if delay == 0:
                intervals.append(dict(policy=name, **daily_bootstrap(final.net)))
                ts = pd.to_datetime(events.timestamp, utc=True).dt.tz_convert('America/New_York')
                entry = (events.before == 0) & (events.after != 0) & (ts.dt.year >= 2024)
                counts = ts.loc[entry].dt.hour.value_counts().reindex(range(24), fill_value=0)
                entry_times.extend(dict(policy=name, hour_ny=h, entries=int(n)) for h, n in counts.items())
        pd.DataFrame(rows).to_csv(OUTPUT/'comparison.csv', index=False)
        print('Completed', name, flush=True)

    fixed_marks, feature_cache = {}, {}
    for hour, p in fixed:
        if hour not in fixed_marks:
            marks = clock_marks(data, hour)
            fixed_marks[hour] = marks
            marks.to_csv(OUTPUT/f'marks_clock{hour:02d}.csv')
            cov = marks.level.notna().groupby(marks.index.year).agg(['count', 'sum', 'mean']).reset_index()
            cov['mode'], cov['hour_ny'] = 'daily_clock', hour
            coverage.append(cov)
        marks = fixed_marks[hour]
        key = (hour, p.kind, p.horizon_days)
        if key not in feature_cache:
            feature_cache[key], _ = daily_features(marks, p.kind, p.horizon_days)
        signal = daily_signal(marks, feature_cache[key], p)
        evaluate(signal, dict(family='daily', kind=p.kind, horizon_hours=p.horizon_days*24,
                              entry_z=p.entry_z, review=f'clock{hour:02d}', hour_ny=hour))
        if hour == 16:
            old = HERE/'sgd_neer_results/multiday'/f'daily_{p.kind}_{p.horizon_days}d_z{p.entry_z:g}_fill0.csv.gz'
            old_frame = pd.read_csv(old, index_col=0, parse_dates=True)
            new_frame = pd.read_csv(OUTPUT/f'daily_{p.name}_fill0.csv.gz', index_col=0, parse_dates=True)
            pd.testing.assert_frame_equal(old_frame, new_frame)
    print('All eight 16:00 controls exactly reproduce prior multi-day PnL.', flush=True)
    marks = rolling_marks(data)
    cov = marks.level.notna().groupby(marks.index.year).agg(['count', 'sum', 'mean']).reset_index()
    cov['mode'], cov['hour_ny'] = 'rolling_grid', np.nan
    coverage.append(cov)
    previous_key, features = None, None
    for p in rolling:
        key = (p.kind, p.horizon_hours)
        if key != previous_key:
            features = rolling_features(marks, *key)
            previous_key = key
            usable = features.z.notna() & marks.level.notna() & marks.vol.notna()
            cov = usable.groupby(marks.index.year).agg(['count', 'sum', 'mean']).reset_index()
            cov['mode'], cov['hour_ny'] = f'rolling_{key[0]}_{key[1]}h_usable', np.nan
            coverage.append(cov)
        signal = rolling_signal(marks, features, p)
        evaluate(signal, dict(family='rolling', kind=p.kind, horizon_hours=p.horizon_hours,
                              entry_z=p.entry_z, review='continuous' if p.review_hour is None else 'clock16',
                              hour_ny=p.review_hour))
    frame = pd.DataFrame(rows)
    metadata = pd.DataFrame(metadata)
    frame = frame.merge(metadata, on='policy', validate='many_to_one')
    frame.to_csv(OUTPUT/'comparison.csv', index=False)
    metadata.to_csv(OUTPUT/'policies.csv', index=False)
    pd.concat(coverage).to_csv(OUTPUT/'coverage.csv', index=False)
    primary = frame[frame.fill_delay_minutes == 0]
    dev = primary[primary.period.isin(['development', 'recent_development', 'validation'])]
    board = dev.pivot(index='policy', columns='period', values='sharpe')
    val = dev[dev.period == 'validation'].set_index('policy')
    board['validation_completed_trades'], board['validation_active_days'] = val.completed_trades, val.active_days
    board['score'] = board[['development', 'recent_development', 'validation']].min(axis=1, skipna=False)
    board['eligible'] = ((board.validation_completed_trades >= 10) & (board.validation_active_days >= 40)
                         & np.isfinite(board.score) & (board.score > 0))
    board = board.sort_values('score', ascending=False)
    board.to_csv(OUTPUT/'leaderboard.csv')
    selected = list(board[board.eligible].head(3).index)
    (OUTPUT/'frozen_selection.json').write_text(json.dumps({'selected': selected, 'selection': protocol['selection']}, indent=2))
    pd.DataFrame(stress).to_csv(OUTPUT/'stress.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUTPUT/'bootstrap.csv', index=False)
    pd.DataFrame(entry_times).to_csv(OUTPUT/'entry_hours.csv', index=False)
    (OUTPUT/'code_manifest.json').write_text(json.dumps([
        {'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths], indent=2))
    (OUTPUT/'run_signature.txt').write_text(signature)
    print('Development-only shortlist:', selected, flush=True)
    result = primary[primary.period == 'evaluation']
    print(result[['policy', 'sharpe', 'gross_sharpe', 'completed_trades']].round(3).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
