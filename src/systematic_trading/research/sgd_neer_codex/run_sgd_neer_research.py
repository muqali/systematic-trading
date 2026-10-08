"""Reproduce the SGD NEER research without executing the exploratory notebook.

From the repository root:
    poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_research.py --phase evaluation
    poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_research.py --phase development

Evaluation uses an already frozen shortlist and never selects on its results.
Only the pre-existing local quote caches are read. No quotes are downloaded.
"""
import argparse
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.resampled_data_loader import get_resampled_quotes, RESAMPLED_DATA_DIR
from research.sgd_neer import DEFAULT_WEIGHTS
from research.sgd_neer_codex.sgd_neer_backtest import (
    Policy, prepare_basket, backtest, metrics, daily_pnl, policy_signal,
    rolling_forecast, component_forecast_signals, block_bootstrap_sharpe,
    mean_reversion_diagnostics,
    signals_to_pairs,
)
from strategy.sgd_neer_strategy import SGDNEERStrategy

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / 'sgd_neer_results'
PAIRS = ['USDSGD', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY']
BASELINE_PARAMS = {
    'past_horizon': pd.Timedelta('120min'), 'future_horizon': pd.Timedelta('30min'),
    'zscore_window': pd.Timedelta('20D'), 'zscore_entry_threshold': 1.,
    'spread_rolling_window': pd.Timedelta('15min'), 'margin_multiple': .5,
    'adjust_to_target': True, 'delay_trade_on_signal': False,
}


def all_policies():
    policies = []
    for name in ['protocol.json', 'extension_protocol.json', 'component_protocol.json']:
        policies.extend(Policy(**p) for p in json.loads((DEFAULT_OUTPUT/name).read_text())['policies'])
    return {p.name: p for p in policies}


def original_baselines():
    snapshot = pd.read_csv(DEFAULT_OUTPUT/'original_index_signal.csv.gz')
    index = pd.to_datetime(snapshot.iloc[:, 0], utc=True).dt.tz_convert('America/New_York')
    signal = pd.Series(snapshot.signal.to_numpy(), index=pd.DatetimeIndex(index), name='index')
    # As loaded before the fix: the last (GBP) cross overwrote the SGD hedge.
    old_loadings = {p: (-DEFAULT_WEIGHTS['GBPUSD'] if p == 'USDSGD' else
                        DEFAULT_WEIGHTS[p] * (1 if p.startswith('USD') else -1)) for p in PAIRS}
    return {
        'legacy_original': pd.DataFrame({p: signal*w for p, w in old_loadings.items()}),
        'legacy_mapping_only': signal,
    }


def make_signal(data, policy, forecasts):
    if policy.name.startswith('components_'):
        pairs = ['EURUSD', 'USDJPY'] if policy.name.endswith('liquid') else None
        return component_forecast_signals(data, policy, pairs)
    fit = pd.DataFrame()
    pred = None
    if policy.kind.startswith('forecast'):
        key = (policy.forecast_horizon, policy.training_weeks, policy.execution_delay)
        if key not in forecasts:
            forecasts[key] = rolling_forecast(data, key[0], key[1], execution_delay=key[2])
        pred, fit = forecasts[key]
    return policy_signal(data, policy, pred), fit


def file_manifest(end_year):
    files = []
    for pair in PAIRS:
        for year in range(2020, end_year+1):
            path = RESAMPLED_DATA_DIR/f'DAT_ASCII_{pair}_M15_{year}.csv'
            if not path.exists():
                raise FileNotFoundError(f'Required local quote cache missing: {path}')
            files.append({'file': str(path), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return files


def run(phase, output, execution_delay=1):
    output.mkdir(parents=True, exist_ok=True)
    end_year = 2023 if phase == 'development' else 2025
    manifest = file_manifest(end_year)
    prices = get_resampled_quotes(PAIRS, dt.date(2020,1,1), dt.date(end_year,12,31), pd.Timedelta('15min'))
    data = prepare_basket(prices)
    audit = dict(data.audit, pairs=PAIRS, loadings=data.loadings.to_dict(),
                 raw_weight_coverage=sum(DEFAULT_WEIGHTS[p] for p in PAIRS)/sum(DEFAULT_WEIGHTS.values()))
    quality = []
    union = pd.DataFrame({p:q.mid for p,q in prices.items()})
    for year, part in union.groupby(union.index.year):
        quality.append({'year':year, 'rows':len(part), 'complete_rows':int(part.notna().all(axis=1).sum()),
                        **{f'{p}_missing':int(part[p].isna().sum()) for p in PAIRS}})
    pd.DataFrame(quality).to_csv(output/f'data_quality_{phase}.csv',index=False)
    (output/f'data_manifest_{phase}.json').write_text(json.dumps({'audit':audit,'files':manifest},indent=2))
    print('Quote audit:', audit, flush=True)
    signals = original_baselines()
    strategy = SGDNEERStrategy(PAIRS, prices, BASELINE_PARAMS)
    signals['legacy_corrected'] = pd.DataFrame(strategy.generate_signals())
    policies = {name: replace(policy, execution_delay=execution_delay)
                for name, policy in all_policies().items()}
    if phase == 'development':
        names = list(policies)
        (output/'run_protocol.json').write_text(json.dumps({
            'execution_delay': execution_delay, 'policies': [asdict(p) for p in policies.values()],
            'development': '2021-2022', 'validation': '2023', 'evaluation': '2024-2025',
            'selection': 'Top three min(development, validation) net Sharpe; at least 40 validation trade days.',
            'evaluation_status': 'Previously inspected in earlier research; this is a follow-up experiment.'
        }, indent=2))
    else:
        freeze_root = DEFAULT_OUTPUT if execution_delay == 1 else output
        freeze = json.loads((freeze_root/'frozen_selection.json').read_text())
        if freeze.get('execution_delay', 1) != execution_delay:
            raise ValueError('Frozen selection and requested execution delay disagree.')
        names = freeze['selected_for_diagnostic_evaluation']
    forecasts = {}
    all_fits = []
    for name in names:
        signal, fits = make_signal(data, policies[name], forecasts)
        signals[name] = signal
        if not fits.empty:
            all_fits.append(fits.assign(policy=name))
        print('Generated', name, flush=True)
    if phase == 'evaluation':
        signals['frozen_ensemble'] = sum(signals_to_pairs(data, signals[n]) for n in names)/len(names)
    rows, stress_rows, daily = [], [], {}
    periods = [('development', '2021', '2022'), ('validation', '2023', '2023')]
    if phase == 'evaluation':
        periods += [('evaluation', '2024', '2025'), ('2024', '2024', '2024'),
                    ('2025', '2025', '2025'), ('all_2021_2025','2021','2025')]
    for name, signal in signals.items():
        bt = backtest(data, signal, delay=execution_delay)
        for period, start, end in periods:
            part = bt.loc[start:end]
            row = dict(policy=name, period=period, execution_delay=execution_delay, **metrics(part.net))
            row.update(gross_sharpe=metrics(part.gross)['sharpe'],
                       trade_days=int((daily_pnl(part.turnover)>0).sum()),
                       avg_daily_turnover=float(daily_pnl(part.turnover).mean()),
                       total_cost=float(part.cost.sum()), total_gross=float(part.gross.sum()),
                       rollover_events_with_exposure=int((part.rollover_exposure>0).sum()))
            if period == 'evaluation':
                row.update({f'bootstrap_{k}':v for k,v in block_bootstrap_sharpe(part.net).items()})
                row['break_even_cost_multiple'] = part.gross.sum()/part.cost.sum() if part.cost.sum()>0 else np.nan
            rows.append(row)
        for col in ['gross','cost','net','turnover','rollover_exposure']:
            daily[f'{name}__{col}'] = daily_pnl(bt[col])
        if phase == 'evaluation':
            for label, delay, cost, slip in [
                ('half_spread_cost',execution_delay,.5,0), ('base',execution_delay,1.,0),
                ('spread_x1.5',execution_delay,1.5,0), ('spread_x2',execution_delay,2.,0),
                ('extra_0.25bp_per_leg',execution_delay,1.,.25),
                ('same_bar_optimistic',0,1.,0), ('two_bar_delay',2,1.,0),
                ('one_bar_delay',1,1.,0),
            ]:
                stressed = backtest(data,signal,delay=delay,cost_multiple=cost,extra_slippage_bps=slip)
                stress_rows.append(dict(policy=name, scenario=label, **metrics(stressed.net.loc['2024':'2025'])))
            financing = bt.net-bt.rollover_exposure*1e-4
            stress_rows.append(dict(policy=name,scenario='1bp_per_roll_exposure',**metrics(financing.loc['2024':'2025'])))
    result = pd.DataFrame(rows)
    result.to_csv(output/f'comparison_{phase}.csv',index=False)
    pd.DataFrame(daily).to_csv(output/f'daily_{phase}.csv.gz',compression='gzip')
    if all_fits:
        pd.concat(all_fits).to_csv(output/f'fits_{phase}.csv',index=False)
    if stress_rows:
        pd.DataFrame(stress_rows).to_csv(output/'stress_evaluation.csv',index=False)
    if phase == 'development':
        vr,reg=mean_reversion_diagnostics(data)
        vr.to_csv(output/'variance_ratio.csv',index=False)
        reg.to_csv(output/'return_regression.csv',index=False)
        if execution_delay != 1:
            trials = result[~result.policy.str.startswith('legacy')]
            board = trials.pivot(index='policy', columns='period', values='sharpe')
            board['validation_trade_days'] = trials.query("period == 'validation'").set_index('policy').trade_days
            board['score'] = board[['development','validation']].min(axis=1, skipna=False)
            eligible = board[(board.validation_trade_days >= 40) & np.isfinite(board.score)].sort_values('score', ascending=False)
            selected = list(eligible.head(3).index)
            if not selected:
                raise ValueError('No eligible policies; final evaluation has no shortlist.')
            (output/'frozen_selection.json').write_text(json.dumps({
                'execution_delay': execution_delay,
                'selected_for_diagnostic_evaluation': selected,
                'positive_in_both_periods': list(eligible[eligible.score > 0].index),
                'selection': 'Descending min(development, validation) net Sharpe; at least 40 validation trade days.',
                'ensemble': 'Equal weight of selected pair-notional vectors, netted before costs.',
                'evaluation_status': 'Follow-up on previously seen evaluation dates; not a new untouched holdout.',
            }, indent=2))
            board.sort_values('score', ascending=False).to_csv(output/'development_leaderboard.csv')
            print('Frozen shortlist:', selected, flush=True)
    print(result.pivot(index='policy',columns='period',values='sharpe').round(3).to_string())
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['development','evaluation'],default='evaluation')
    parser.add_argument('--execution-delay',type=int,choices=[0,1,2],default=1)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    output = args.output or (DEFAULT_OUTPUT if args.execution_delay == 1
                             else DEFAULT_OUTPUT/f'delay_{args.execution_delay}')
    run(args.phase,output,args.execution_delay)
