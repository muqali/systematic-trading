"""Holding-time sensitivity and a declared equilibrium parameter grid.

Run from the repository root with poetry run python <this file>.
Selection uses 2017--2023 only; 2024--2025 is previously inspected evaluation.
Every configuration regenerates its complete causal entry/exit path.
"""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE as MINUTE_CACHE, daily_metrics
from research.sgd_neer_codex.sgd_neer_timing import RollingPolicy, rolling_marks, rolling_features, rolling_signal
from research.sgd_neer_codex.sgd_neer_subset import (
    cross_structure, covariance_at_entries, component_deviations, subset_targets, PairAccountant,
)
from research.sgd_neer_codex.sgd_neer_multiday import daily_bootstrap
from research.sgd_neer_codex.run_sgd_neer_subset import paired_interval

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/"sgd_neer_results/holding_time"
HORIZONS = [12, 24, 48, 72]
THRESHOLDS = [1., 1.5, 2.]
HOLDING = [48, 72, 120, 240, 480, "unlimited"]
VARIANTS = [("aligned", "risk_matched"), ("aligned", "retained"),
            ("aligned", "normalized"), ("full", "retained")]
PERIODS = {"development": ("2017", "2022"), "recent_development": ("2021", "2022"),
           "validation": ("2023", "2023"), "evaluation": ("2024", "2025"),
           "weak_2019_2021": ("2019", "2021"), "early_2016_2018": ("2016", "2018"),
           "later_2022_2025": ("2022", "2025"), "all_2017_2025": ("2017", "2025"),
           "2024": ("2024", "2024"), "2025": ("2025", "2025")}
SELECTION_PERIODS = ["development", "recent_development", "validation"]
BASELINE = "equilibrium_24h_z1.5_hold48h__aligned__risk_matched"


def policies():
    out = []
    for horizon in HORIZONS:
        limits = list(dict.fromkeys([2*horizon, *HOLDING]))
        for threshold in THRESHOLDS:
            for holding in limits:
                label = "unlimited" if holding == "unlimited" else f"{holding}h"
                out.append(RollingPolicy(f"equilibrium_{horizon}h_z{threshold:g}_hold{label}",
                                        "equilibrium", horizon, entry_z=threshold,
                                        max_holding_hours=holding))
    return out


def read_daily(name, delay):
    return pd.read_csv(OUTPUT/"daily"/f"{name}_fill{delay}.csv.gz", index_col=0, parse_dates=True)


def event_times(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_convert("America/New_York")


def summarize(daily, events, name, delay, period_names):
    ts = event_times(events.timestamp)
    rows = []
    for period in period_names:
        start, end = PERIODS[period]
        part = daily.loc[start:end]
        fills = events.loc[(ts.year >= int(start)) & (ts.year <= int(end))]
        row = dict(policy=name, fill_delay_minutes=delay, period=period, **daily_metrics(part.net))
        row.update(gross_sharpe=daily_metrics(part.gross)["sharpe"],
                   total_gross=part.gross.sum(), total_cost=part.cost.sum(),
                   total_net=part.net.sum(), turnover=part.turnover.sum(),
                   annual_cost=part.cost.mean()*252,
                   active_days=int((part.active_minutes > 0).sum()),
                   completed_trades=int(((fills.before != 0) & (fills.after == 0)).sum()),
                   gross_exposure_minutes=part.gross_exposure_minutes.sum())
        rows.append(row)
    return rows


def ranking(development, metadata):
    primary = development.query("fill_delay_minutes == 0")
    board = primary.pivot(index="policy", columns="period", values="sharpe")
    board["score"] = board[SELECTION_PERIODS].min(axis=1, skipna=False)
    validation = primary.query("period == 'validation'").set_index("policy")
    board["validation_completed_trades"] = validation.completed_trades
    board["validation_active_days"] = validation.active_days
    board["eligible"] = ((board.score > 0) & (board.validation_completed_trades >= 10)
                         & (board.validation_active_days >= 40))
    return board.join(metadata.set_index("policy")).sort_values("score", ascending=False)


def run():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for directory in ["signals", "targets", "decisions", "daily", "trades"]:
        (OUTPUT/directory).mkdir(exist_ok=True)
    parents = policies()
    protocol = dict(index_policies=[asdict(p) for p in parents], variants=VARIANTS,
                    signals=len(parents), configurations=len(parents)*len(VARIANTS),
                    periods=PERIODS, selection_periods=SELECTION_PERIODS,
                    ranking="Maximise minimum same-quote net Sharpe across 2017-2022, 2021-2022, 2023; "
                            "positive minimum, >=10 validation completed trades, >=40 validation active days. "
                            "No improvement-over-parent requirement: this grid compares full and subset strategies directly.",
                    holding="Scheduled market hours; weekend closure excluded, missing market minutes retained. "
                            "None in the research policy retains 2H default; explicit unlimited disables time exits. "
                            "Near/cross-mean exit at 0.25z remains; constant entry allocation until exit. "
                            "Regenerate all entries, since prolonged holdings block later opportunities.",
                    controls="Existing 24h/1.5z/48h aligned risk-matched and full-index results must reproduce at delays 0/1.",
                    unchanged="63-day z scale, 20-day risk half-life, 40-day history, 24h return covariance; "
                              "5% entry volatility target, max 3 cross units; half-deviation edge >1.5 full-index spreads.",
                    execution="Same quote and next common valid quote; daily mark-to-market includes held losses and gaps; "
                              "recorded bid/ask costs and final liquidation.",
                    financing="Excluded at user's request, including during extended holding periods.",
                    evaluation="2024-2025 already examined in earlier work; exploratory, not a fresh holdout. "
                               "Freeze pre-2024 choices before computing evaluation rankings. Report all configurations.")
    (OUTPUT/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    source_paths = [Path(__file__), HERE/"sgd_neer_timing.py", HERE/"sgd_neer_subset.py",
                    HERE/"sgd_neer_multiday.py", HERE/"sgd_neer_backtest.py", HERE.parent/"sgd_neer.py"]
    source_manifest = [{"file":str(p), "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]
    signature = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    signature_path = OUTPUT/"run_signature.txt"
    resume = signature_path.exists() and signature_path.read_text() == signature
    # Inputs are local prepared quotes; a changed quote cache invalidates reuse.
    with (MINUTE_CACHE/"data.pkl").open("rb") as stream:
        data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    hash_path = OUTPUT/"prepared_data_sha256.txt"
    resume = resume and hash_path.exists() and hash_path.read_text() == data_hash
    signature_path.write_text(signature)
    hash_path.write_text(data_hash)
    (OUTPUT/"code_manifest.json").write_text(json.dumps(source_manifest, indent=2)+"\n")
    data = pd.read_pickle(MINUTE_CACHE/"data.pkl")
    marks = rolling_marks(data)
    market_index = marks.index
    signals = {}
    print(f"Declared {len(parents)} index rules / {len(parents)*len(VARIANTS)} portfolios", flush=True)
    for horizon in HORIZONS:
        features = rolling_features(marks, "equilibrium", horizon)
        for policy in [p for p in parents if p.horizon_hours == horizon]:
            path = OUTPUT/"signals"/f"{policy.name}.pkl"
            if resume and path.exists():
                signal = pd.read_pickle(path)
            else:
                signal = rolling_signal(marks, features, policy)
                signal.to_pickle(path)
            signals[policy.name] = signal
            print(f"Signal {policy.name}: {(signal != 0).sum()} entries; {time.monotonic()-started:.0f}s", flush=True)
        del features
    entries = pd.DatetimeIndex(sorted({t for signal in signals.values() for t in signal[signal != 0].index}))
    matrix, weights = cross_structure(list(data.mids.columns))
    levels = np.log(data.mids).dot(matrix.T).reindex(market_index)
    print(f"Computing prior entry covariance at {len(entries)} distinct entries", flush=True)
    covariance = covariance_at_entries(levels, entries)
    deviations = {}
    for horizon in HORIZONS:
        full = component_deviations(levels, "equilibrium", horizon)
        deviations[horizon] = full.reindex(entries)
    del levels, full, marks
    gc.collect()
    accountant = PairAccountant(data)
    metadata, development, hold_stats = [], [], []
    controls_verified = []
    for k, policy in enumerate(parents):
        signal = signals[policy.name]
        entry_times = signal[signal != 0].index
        d = deviations[policy.horizon_hours].reindex(entry_times)
        spreads = data.spreads.reindex(entry_times).dot(matrix.abs().T)
        cov = covariance[entries.get_indexer(entry_times)]
        for rule, sizing in VARIANTS:
            name = f"{policy.name}__{rule}__{sizing}"
            target_path, decision_path = OUTPUT/"targets"/f"{name}.pkl", OUTPUT/"decisions"/f"{name}.csv.gz"
            if resume and target_path.exists() and decision_path.exists():
                targets = pd.read_pickle(target_path)
                decisions = pd.read_csv(decision_path)
            else:
                targets, decisions = subset_targets(signal, d, spreads, cov, matrix, weights, rule, sizing)
                targets.to_pickle(target_path)
                decisions.to_csv(decision_path, index=False, compression="gzip")
            metadata.append(dict(policy=name, horizon_hours=policy.horizon_hours, entry_z=policy.entry_z,
                                 holding_hours=policy.max_holding_hours, rule=rule, sizing=sizing))
            for delay in [0, 1]:
                daily_path = OUTPUT/"daily"/f"{name}_fill{delay}.csv.gz"
                trades_path = OUTPUT/"trades"/f"{name}_fill{delay}.csv.gz"
                if resume and daily_path.exists() and trades_path.exists():
                    daily = read_daily(name, delay)
                    events = pd.read_csv(trades_path)
                else:
                    daily, events = accountant.run(targets, delay=delay)
                    daily.to_csv(daily_path, compression="gzip")
                    events.to_csv(trades_path, index=False, compression="gzip")
                if policy.horizon_hours == 24 and policy.entry_z == 1.5 and policy.max_holding_hours == 48:
                    original = f"equilibrium_24h_z1.5__{rule}__{sizing}"
                    old = pd.read_csv(HERE/"sgd_neer_results/subset"/f"daily_{original}_fill{delay}.csv.gz",
                                      index_col=0, parse_dates=True)
                    np.testing.assert_allclose(daily[old.columns], old, atol=1e-10)
                    old_targets = pd.read_pickle(HERE.parents[3]/"tmp/sgd_neer_subset"/f"{original}.pkl")
                    pd.testing.assert_frame_equal(targets, old_targets, atol=1e-9, rtol=1e-9)
                    controls_verified.append(dict(policy=name, delay=delay, max_daily_error=float((daily-old).abs().max().max())))
                development.extend(summarize(daily, events, name, delay, SELECTION_PERIODS))
                if delay == 0:
                    times = event_times(events.timestamp)
                    opens = np.flatnonzero(events.before.eq(0) & events.after.gt(0))
                    closes = np.flatnonzero(events.after.eq(0) & events.before.gt(0))
                    assert len(opens) == len(closes) and np.all(closes == opens+1)
                    hours = (market_index.get_indexer(times[closes])-market_index.get_indexer(times[opens]))/60
                    for period, (start, end) in PERIODS.items():
                        keep = (times[opens].year >= int(start)) & (times[opens].year <= int(end))
                        h = hours[keep]
                        holding = policy.max_holding_hours
                        hold_stats.append(dict(policy=name, period=period, trades=len(h),
                                               mean_hours=h.mean() if len(h) else np.nan,
                                               p95_hours=np.quantile(h,.95) if len(h) else np.nan,
                                               max_hours=h.max() if len(h) else np.nan,
                                               timed_exit_share=(np.mean(h >= holding) if len(h) and holding != "unlimited" else 0.)))
        pd.DataFrame(development).to_csv(OUTPUT/"comparison_development.csv", index=False)
        print(f"Accounted {k+1}/{len(parents)} index rules; {time.monotonic()-started:.0f}s", flush=True)
    meta = pd.DataFrame(metadata)
    meta.to_csv(OUTPUT/"policies.csv", index=False)
    pd.DataFrame(hold_stats).to_csv(OUTPUT/"holding_statistics.csv", index=False)
    board = ranking(pd.DataFrame(development), meta)
    board.to_csv(OUTPUT/"leaderboard_development.csv")
    eligible = board[board.eligible]
    def choose(frame):
        return list(frame.head(3).index)
    fixed = eligible[(eligible.horizon_hours == 24) & (eligible.entry_z == 1.5)
                     & (eligible.rule == "aligned") & (eligible.sizing == "risk_matched")]
    choices = dict(primary=choose(eligible), aligned_risk_matched=choose(eligible[(eligible.rule == "aligned") & (eligible.sizing == "risk_matched")]),
                   holding_only=choose(fixed), fixed_48h=choose(eligible[eligible.holding_hours == 48]),
                   unlimited=choose(eligible[eligible.holding_hours == "unlimited"]))
    frozen = dict(selected=choices, baseline=BASELINE, selection=protocol["ranking"],
                  evaluation_status=protocol["evaluation"])
    (OUTPUT/"frozen_selection.json").write_text(json.dumps(frozen, indent=2)+"\n")
    print("FROZEN PRE-2024 CHOICES", json.dumps(choices), flush=True)
    (OUTPUT/"verification.json").write_text(json.dumps(controls_verified, indent=2)+"\n")
    del accountant, data, covariance, deviations
    gc.collect()

    # Evaluation begins only after the above ranking and choices are frozen.
    rows, annual, stress, intervals = [], [], [], []
    selected = set([BASELINE, *(name for group in choices.values() for name in group)])
    for name in meta.policy:
        for delay in [0, 1]:
            daily = read_daily(name, delay)
            events = pd.read_csv(OUTPUT/"trades"/f"{name}_fill{delay}.csv.gz")
            rows.extend(summarize(daily, events, name, delay, list(PERIODS)))
            for year, part in daily.groupby(daily.index.year):
                annual.append(dict(policy=name, delay=delay, year=year, **daily_metrics(part.net),
                                   gross=part.gross.sum(), cost=part.cost.sum(), net=part.net.sum()))
            final = daily.loc["2024":"2025"]
            for scenario, mult, slip in [("base",1.,0.),("spread_x1.5",1.5,0.),("spread_x2",2.,0.),("extra_0.25bp_per_leg",1.,.25)]:
                pnl = final.gross-mult*final.cost-slip/1e4*final.turnover
                stress.append(dict(policy=name, delay=delay, scenario=scenario, **daily_metrics(pnl)))
            if name in selected and delay == 0:
                reference = read_daily(BASELINE, 0).loc["2024":"2025", "net"]
                intervals.append(dict(policy=name, **daily_bootstrap(final.net), **paired_interval(final.net, reference)))
    comparison = pd.DataFrame(rows).merge(meta, on="policy", validate="many_to_one")
    comparison.to_csv(OUTPUT/"comparison.csv", index=False)
    pd.DataFrame(annual).to_csv(OUTPUT/"annual_pnl.csv", index=False)
    pd.DataFrame(stress).to_csv(OUTPUT/"stress.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUTPUT/"bootstrap.csv", index=False)
    top_final = comparison.query("period == 'evaluation' and fill_delay_minutes == 0").sort_values("sharpe", ascending=False)
    top_final.to_csv(OUTPUT/"leaderboard_evaluation_retrospective.csv", index=False)
    print("Retrospective evaluation leaders (not the selection rule):\n",top_final[["policy","sharpe","max_drawdown","annual_vol"]].head(10).round(4).to_string(index=False),flush=True)
    print(f"Complete in {time.monotonic()-started:.0f}s; {OUTPUT}",flush=True)


if __name__ == "__main__":
    run()
