"""Attribute index extremes and compare frozen constituent subsets.

Run: poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_subset.py
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.sgd_neer_timing import RollingPolicy, rolling_marks, rolling_features, rolling_signal
from research.sgd_neer_codex.sgd_neer_subset import (
    cross_structure, component_deviations, covariance_at_entries, subset_targets, PairAccountant,
)
from research.sgd_neer_codex.sgd_neer_multiday import daily_bootstrap
from research.sgd_neer_codex.run_sgd_neer_multiday import summarize
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE as MINUTE_CACHE, PERIODS, daily_metrics

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'sgd_neer_results/subset'
CACHE = HERE.parents[3]/'tmp/sgd_neer_subset'


def triggers():
    return [RollingPolicy(f'rolling_{kind}_{h}h_z{z:g}_continuous', kind, h, entry_z=z)
            for kind, h, z in [('equilibrium', 24, 1.5), ('equilibrium', 24, 2.),
                               ('equilibrium', 72, 1.5), ('return', 24, 2.)]]


def configurations():
    return [('full', 'retained')]+[(rule, sizing)
        for rule in ['aligned', 'top1', 'top2', 'coverage80', 'cost_top2']
        for sizing in ['retained', 'normalized', 'risk_matched']]


def paired_interval(candidate, reference, draws=2000, block=20):
    assert candidate.index.equals(reference.index)
    rng = np.random.default_rng(1941)
    n = len(candidate)
    starts = rng.integers(0, n, (draws, int(np.ceil(n/block))))
    indices = ((starts[..., None]+np.arange(block)) % n).reshape(draws, -1)[:, :n]
    a, b = candidate.to_numpy()[indices], reference.to_numpy()[indices]
    sa, sb = a.std(axis=1, ddof=1), b.std(axis=1, ddof=1)
    keep = (sa > 0) & (sb > 0)
    difference = np.sqrt(252)*(a[keep].mean(axis=1)/sa[keep]-b[keep].mean(axis=1)/sb[keep])
    quantiles = np.quantile(difference, [.025, .5, .975]) if len(difference) else [np.nan]*3
    return dict(zip(['delta_low', 'delta_median', 'delta_high'], quantiles),
                nondegenerate_draws=int(keep.sum()), paired_draws=draws)


def attribution_outcomes(data, signal, deviations, matrix, weights):
    """Descriptive component outcomes from an index entry to its original exit.

    Future outcomes are diagnostics only; none are passed to subset selection.
    Each synthetic cross is charged its own two-leg entry and exit spread.
    """
    index = data.mids.index
    logs, spreads = np.log(data.mids.to_numpy()), data.spreads.to_numpy()
    rows = []
    for k, (timestamp, q) in enumerate(signal[signal != 0].items()):
        entry_event = signal.index.get_loc(timestamp)
        end = signal.index[entry_event+1] if entry_event+1 < len(signal) else index[-1]
        a, b = index.searchsorted(timestamp), min(index.searchsorted(end), len(index)-1)
        direction = np.sign(q)
        d = deviations.iloc[k].to_numpy()
        signed = -direction*weights.to_numpy()*d
        order = np.argsort(-signed, kind='stable')
        ranks = np.empty(len(order), dtype=int); ranks[order] = np.arange(1, len(order)+1)
        cross_return = matrix.to_numpy()@(logs[b]-logs[a])
        loadings = direction*matrix.to_numpy()
        costs = (loadings*np.log1p(np.sign(loadings)*spreads[a]*.5)-
                 loadings*np.log1p(-np.sign(loadings)*spreads[b]*.5)).sum(axis=1)
        for j, ccy in enumerate(weights.index):
            rows.append(dict(timestamp=timestamp, exit_timestamp=index[b], currency=ccy,
                             weight=weights.iloc[j], deviation=d[j], contribution=weights.iloc[j]*d[j],
                             aligned=bool(signed[j] > 0), contribution_rank=int(ranks[j]),
                             gross_fade_return=direction*cross_return[j], cost=costs[j],
                             net_fade_return=direction*cross_return[j]-costs[j],
                             weighted_gross_pnl=abs(q)*weights.iloc[j]*direction*cross_return[j],
                             weighted_cost=abs(q)*weights.iloc[j]*costs[j]))
    return pd.DataFrame(rows)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    parents, variants = triggers(), configurations()
    protocol = {
        'index_triggers': [asdict(p) for p in parents], 'subset_variants': variants,
        'count': len(parents)*len(variants), 'periods': PERIODS,
        'source': str(MINUTE_CACHE/'data.pkl'), 'freq': '1 minute',
        'trigger': 'Use exactly the existing full-index entries, spread hurdle, size and exits. '
                   'Subsets cannot create new entries; selections are frozen until the index exit.',
        'decomposition': 'Construct SGD/USD, SGD/EUR, SGD/AUD, SGD/GBP and SGD/JPY log levels. '
                         'Apply the same causal index deviation operator to each; sum(w_i*d_i) equals index deviation.',
        'selection': 'Only same-sign index contributors are eligible. aligned: all positive contributors; '
                     'top1/top2: largest weighted contributions; coverage80: smallest ranked set explaining '
                     '80% of positive contributions, not 80% of the potentially offset net total. '
                     'cost_top2: at most two contributors with half signed deviation >1.5 synthetic full spreads, '
                     'ranked by weighted contribution / weighted synthetic spread. The half-reversion edge is heuristic.',
        'sizing': 'retained keeps original component weights; normalized rescales selected weights to sum to one; '
                  'risk_matched uses normalized weights and scales entry size to the full portfolio prior covariance risk, '
                  'capped at three aggregate SGD-cross units. Covariance uses lagged empirical 24-market-hour returns, '
                  '20-market-day EWM half life and 40-market-day minimum history. No future realized-vol scaling.',
        'hedges': 'Translate each selected SGD/X to actual USD-quoted legs and aggregate USDSGD before costing. '
                  'Keeping total SGD exposure fixed keeps the USDSGD hedge unchanged. Few crosses need not cost less.',
        'execution': 'Same quote primary; one subsequent common valid quote sensitivity, retaining all gap PnL. '
                     'Minute ledger telescopes returns between fills/day boundaries; checked against dense five-leg engine.',
        'financing': 'No forward-point adjustment, explicitly requested by user.',
        'ranking': 'Minimum net Sharpe across development 2017-2022, recent 2021-2022 and validation 2023. '
                   'Subset shortlist requires positive Sharpe AND improvement over its full-index parent in all three '
                   'periods, >=10 validation completed trades and >=40 active days. Rank minimum Sharpe improvement; top three.',
        'evaluation': '2024-2025 already inspected previously. All 64 declared results disclosed. Exploratory follow-up, '
                      'not an untouched holdout. Attribution outcomes and final-period scores do not influence selection.',
        'coverage': 'Same common-five-pair quote grid for all portfolios, retaining outages and skipping weekend closure. '
                    'No added quote fill; cached inputs may already be filled by exporter up to ten minutes.',
    }
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    data = pd.read_pickle(MINUTE_CACHE/'data.pkl')
    assert data.freq == pd.Timedelta(minutes=1)
    accountant = PairAccountant(data)
    matrix, weights = cross_structure(list(data.mids.columns))
    matrix.to_csv(OUTPUT/'cross_to_pair_loadings.csv')
    weights.to_csv(OUTPUT/'component_weights.csv')
    marks = rolling_marks(data)
    signals, features = {}, {}
    for policy in parents:
        key = (policy.kind, policy.horizon_hours)
        if key not in features:
            features[key] = rolling_features(marks, *key)
        signals[policy.name] = rolling_signal(marks, features[key], policy)
        old_signal = pd.read_pickle(HERE.parents[3]/'tmp/sgd_neer_timing'/f'{policy.name}.pkl')
        pd.testing.assert_series_equal(signals[policy.name], old_signal)
    entries = pd.DatetimeIndex(sorted(set(t for s in signals.values() for t in s[s != 0].index)))
    levels = np.log(data.mids).dot(matrix.T).reindex(marks.index)
    print('Reproduced all index signals; computing entry covariance for', len(entries), 'events.', flush=True)
    covariance = covariance_at_entries(levels, entries)
    deviation_cache = {}
    for key in features:
        full = component_deviations(levels, *key)
        deviation_cache[key] = full.reindex(entries)
        np.testing.assert_allclose(full.mul(weights).sum(axis=1, min_count=len(weights)),
                                   -2*features[key].edge, atol=1e-11)
    del full, levels, features, marks
    print('Component deviations add to index deviation; all risk estimates use preceding quotes.', flush=True)
    rows, metadata, stresses, uncertainty, outcome_frames = [], [], [], [], []
    for policy in parents:
        signal = signals[policy.name]
        trigger = f'{policy.kind}_{policy.horizon_hours}h_z{policy.entry_z:g}'
        entry_times = signal[signal != 0].index
        d = deviation_cache[(policy.kind, policy.horizon_hours)].reindex(entry_times)
        cross_spreads = data.spreads.reindex(entry_times).dot(matrix.abs().T)
        cov = covariance[entries.get_indexer(entry_times)]
        outcomes = attribution_outcomes(data, signal, d, matrix, weights)
        outcomes['trigger'] = trigger
        outcome_frames.append(outcomes)
        baseline = {}
        for rule, sizing in variants:
            name = f'{trigger}__{rule}__{sizing}'
            targets, decisions = subset_targets(signal, d, cross_spreads, cov, matrix, weights, rule, sizing)
            targets.to_pickle(CACHE/f'{name}.pkl')
            decisions.to_csv(OUTPUT/f'decisions_{name}.csv.gz', index=False, compression='gzip')
            metadata.append(dict(policy=name, trigger=trigger, rule=rule, sizing=sizing,
                                 kind=policy.kind, horizon_hours=policy.horizon_hours, entry_z=policy.entry_z))
            for delay in [0, 1]:
                daily, events = accountant.run(targets, delay)
                daily.to_csv(OUTPUT/f'daily_{name}_fill{delay}.csv.gz', compression='gzip')
                events.to_csv(OUTPUT/f'trades_{name}_fill{delay}.csv.gz', index=False, compression='gzip')
                if rule == 'full':
                    baseline[delay] = daily
                    old = pd.read_csv(HERE/'sgd_neer_results/timing'/f'daily_{policy.name}_fill{delay}.csv.gz',
                                      index_col=0, parse_dates=True)
                    np.testing.assert_allclose(daily[old.columns], old, atol=1e-10)
                summary = summarize(daily, events, name, delay, list(PERIODS))
                for row in summary:
                    start, end = PERIODS[row['period']]
                    part = daily.loc[start:end]
                    parent = baseline[delay].loc[start:end]
                    active = decisions[(decisions.timestamp.dt.year >= int(start)) &
                                       (decisions.timestamp.dt.year <= int(end))]
                    row.update(parent_sharpe=daily_metrics(parent.net)['sharpe'],
                               delta_sharpe=row['sharpe']-daily_metrics(parent.net)['sharpe'],
                               cost_ratio=part.cost.sum()/parent.cost.sum(),
                               gross_vol=float(part.gross.std()*np.sqrt(252)),
                               annual_cost=float(part.cost.mean()*252),
                               gross_exposure_ratio=part.gross_exposure_minutes.sum()/parent.gross_exposure_minutes.sum(),
                               usdsgd_cost_share=part.cost_USDSGD.sum()/part.cost.sum() if part.cost.sum() else np.nan,
                               mean_selected_count=float(active.selected_count.mean()),
                               mean_selected_weight=float(active.selected_weight.mean()),
                               mean_sgd_exposure_ratio=float(active.retained_sgd_exposure.mean()),
                               mean_exante_risk_ratio=float(active.exante_risk_ratio.mean()),
                               risk_cap_share=float(active.risk_cap_hit.mean()))
                rows.extend(summary)
                final, reference = daily.loc['2024':'2025'], baseline[delay].loc['2024':'2025']
                for label, mult, slip in [('base',1.,0.), ('spread_x2',2.,0.), ('extra_0.25bp_per_leg',1.,.25)]:
                    pnl = final.gross-final.cost*mult-final.turnover*slip/1e4
                    stresses.append(dict(policy=name, fill_delay_minutes=delay, scenario=label, **daily_metrics(pnl)))
                if delay == 0:
                    uncertainty.append(dict(policy=name, **daily_bootstrap(final.net),
                                            **paired_interval(final.net, reference.net)))
            print('Completed', name, flush=True)
            pd.DataFrame(rows).to_csv(OUTPUT/'comparison.csv', index=False)
    frame = pd.DataFrame(rows).merge(pd.DataFrame(metadata), on='policy', validate='many_to_one')
    frame.to_csv(OUTPUT/'comparison.csv', index=False)
    pd.DataFrame(metadata).to_csv(OUTPUT/'policies.csv', index=False)
    pd.DataFrame(stresses).to_csv(OUTPUT/'stress.csv', index=False)
    pd.DataFrame(uncertainty).to_csv(OUTPUT/'bootstrap.csv', index=False)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    outcomes.to_csv(OUTPUT/'component_outcomes.csv.gz', index=False, compression='gzip')
    primary = frame.query('fill_delay_minutes == 0')
    dev = primary[primary.period.isin(['development','recent_development','validation'])]
    board = dev.pivot(index='policy', columns='period', values='sharpe')
    delta = dev.pivot(index='policy', columns='period', values='delta_sharpe')
    board['score'] = board.min(axis=1, skipna=False)
    board['improvement_score'] = delta.min(axis=1, skipna=False)
    val = dev.query("period == 'validation'").set_index('policy')
    board['validation_completed_trades'], board['validation_active_days'] = val.completed_trades, val.active_days
    board['eligible'] = ((board.score > 0) & (board.improvement_score > 0) &
                         (board.validation_completed_trades >= 10) & (board.validation_active_days >= 40) &
                         (val.rule != 'full'))
    board = board.sort_values('improvement_score', ascending=False)
    board.to_csv(OUTPUT/'leaderboard.csv')
    selected = list(board[board.eligible].head(3).index)
    (OUTPUT/'frozen_selection.json').write_text(json.dumps({'selected': selected, 'selection': protocol['ranking']}, indent=2))
    (OUTPUT/'data_manifest.json').write_bytes((HERE/'sgd_neer_results/minute/data_manifest.json').read_bytes())
    paths = [Path(__file__), HERE/'sgd_neer_subset.py', HERE/'sgd_neer_timing.py', HERE/'sgd_neer_multiday.py',
             HERE.parent/'sgd_neer.py', HERE/'sgd_neer_backtest.py', HERE/'run_sgd_neer_multiday.py', HERE/'run_sgd_neer_minute.py']
    (OUTPUT/'code_manifest.json').write_text(json.dumps([
        {'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths], indent=2))
    with (MINUTE_CACHE/'data.pkl').open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    (OUTPUT/'prepared_data_sha256.txt').write_text(digest)
    print('Development-only subset shortlist:', selected, flush=True)
    print(primary.query("period == 'evaluation'")[['policy','sharpe','delta_sharpe','cost_ratio','mean_exante_risk_ratio']].round(3).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
