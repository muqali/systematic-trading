"""20% tighter spreads for the notebook's 24h/1.5z aligned strategy.

Run from the repository root with poetry run python <this file>. Uses the
prepared one-minute quote cache, starting feature history in 2017 to match
sgd_neer.ipynb as inspected on 2026-09-27. Does not modify source quotes.
"""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE, daily_metrics
from research.sgd_neer_codex.sgd_neer_backtest import daily_pnl, prepare_basket
from research.sgd_neer_codex.sgd_neer_subset import PairAccountant
from strategy.sgd_neer_equilibrium_strategy import SGDNEEREquilibriumStrategy
from trading.aggressive_trader import AggressiveTrader

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "sgd_neer_results" / "spread_sensitivity"
FACTOR = 0.8
PARAMS = {
    "freq": pd.Timedelta(minutes=1),
    "equilibrium_half_life": pd.Timedelta(hours=24),
    "equilibrium_min_history": None,
    "zscore_half_life": pd.Timedelta(days=63),
    "zscore_min_history": pd.Timedelta(days=40),
    "zscore_entry_threshold": 1.5,
    "zscore_exit_threshold": 0.25,
    "max_holding_period": None,
    "reversion_fraction": 0.5,
    "margin_multiple": 1.5,
    "volatility_horizon": pd.Timedelta(hours=24),
    "volatility_half_life": pd.Timedelta(days=20),
    "volatility_min_history": pd.Timedelta(days=40),
    "vol_target": 0.05,
    "max_units": 3.0,
    "constituent_selection": "aligned",
    "sizing_mode": "risk_matched",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def verify_trader(strategy, daily, delay):
    """Check all daily PnL against the notebook's trader, one leg at a time."""
    result = daily[["gross", "net"]] * 0.0
    for pair, quotes in strategy.backtest_prices.items():
        signal = strategy.common_signals[pair].copy()
        signal.iloc[-(delay + 1):] = 0.0
        trader = AggressiveTrader([pair], {pair: quotes}, {pair: signal},
                                  execute_on_next_price_tick=bool(delay))
        part = daily_pnl(pd.DataFrame({
            "gross": trader.generate_gross_pnl()[pair],
            "net": trader.generate_net_pnl()[pair],
        }))
        result += part
        del trader, signal, part
    error = float(np.abs(result - daily[["gross", "net"]]).to_numpy().max())
    np.testing.assert_allclose(result, daily[["gross", "net"]], atol=1e-10, rtol=1e-9)
    return error


def summarize(daily, events, scenario, delay):
    periods = {"2017-2025": ("2017", "2025"), "2024-2025": ("2024", "2025")}
    periods.update({str(year): (str(year), str(year)) for year in range(2017, 2026)})
    rows = []
    for period, (start, end) in periods.items():
        part = daily.loc[start:end]
        net, gross = daily_metrics(part.net), daily_metrics(part.gross)
        entries = events[(events.before == 0) & (events.after > 0)]
        entry_dates = (pd.DatetimeIndex(entries.timestamp).tz_convert("America/New_York")
                       .tz_localize(None) + pd.Timedelta(hours=7)).normalize()
        count = int(((entry_dates.year >= int(start)) & (entry_dates.year <= int(end))).sum())
        rows.append(dict(scenario=scenario, delay=delay, period=period,
                         gross_sharpe=gross["sharpe"], net_sharpe=net["sharpe"],
                         gross_pnl=float(part.gross.sum()), spread_cost=float(part.cost.sum()),
                         net_pnl=float(part.net.sum()), net_annual_pnl=net["annual_pnl"],
                         net_annual_vol=net["annual_vol"], max_drawdown=net["max_drawdown"],
                         entries=count, observed_days=len(part)))
    return rows


def save_run(accountant, targets, scenario, delay, rows):
    daily, events = accountant.run(targets, delay=delay)
    daily.to_csv(OUTPUT / f"{scenario}_delay{delay}_daily.csv.gz", compression="gzip")
    events.to_csv(OUTPUT / f"{scenario}_delay{delay}_events.csv.gz", index=False, compression="gzip")
    rows.extend(summarize(daily, events, scenario, delay))
    return daily, events


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "data.pkl"
    print("Loading prepared minute quotes; discarding pre-2017 history.", flush=True)
    cached = pd.read_pickle(cache_path)
    start = pd.Timestamp("2017-01-01", tz="US/Eastern")
    end = pd.Timestamp("2025-12-31 17:00", tz="US/Eastern")
    keep = (cached.mids.index >= start) & (cached.mids.index < end)
    mids, spreads = cached.mids.loc[keep], cached.spreads.loc[keep]
    prices = {pair: pd.DataFrame({"mid": mids[pair],
                                  "bid": mids[pair] - mids[pair] * spreads[pair] / 2,
                                  "ask": mids[pair] + mids[pair] * spreads[pair] / 2})
              for pair in mids}
    del cached, mids, spreads
    gc.collect()
    audit = {"spread_factor": FACTOR, "start_inclusive": str(start), "end_exclusive": str(end),
             "parameters": PARAMS, "financing": False,
             "cache_sha256": sha256(cache_path),
             "notebook_sha256": sha256(HERE.parent / "sgd_neer.ipynb"),
             "source_sha256": {str(p.relative_to(HERE.parents[3])): sha256(p) for p in [
                 Path(__file__).resolve(), HERE.parents[1] / "strategy/sgd_neer_equilibrium_strategy.py",
                 HERE / "sgd_neer_subset.py", HERE / "sgd_neer_backtest.py",
                 HERE.parents[1] / "trading/aggressive_trader.py"]}}

    print("Generating baseline with the public strategy constructor.", flush=True)
    strategy = SGDNEEREquilibriumStrategy(list(prices), prices, PARAMS)
    strategy.generate_signals()
    targets = strategy.pair_targets.copy()
    baseline_entries = strategy.entry_details.index.copy()
    targets.to_pickle(OUTPUT / "baseline_targets.pkl")
    strategy.entry_details.to_csv(OUTPUT / "baseline_entries.csv.gz", compression="gzip")
    mid_feature_hash = frame_digest(strategy.features.drop(columns="spread"))
    component_hash = frame_digest(strategy.component_deviations)
    data = prepare_basket(strategy.backtest_prices, freq="1min")
    audit.update(quote_rows=len(data.mids), baseline_entries=len(baseline_entries),
                 mid_feature_sha256=mid_feature_hash, component_deviations_sha256=component_hash)
    accountant = PairAccountant(data)
    rows, baseline = [], {}
    for delay in [0, 1]:
        baseline[delay] = save_run(accountant, targets, "baseline", delay, rows)
        audit[f"baseline_trader_max_error_delay{delay}"] = verify_trader(strategy, baseline[delay][0], delay)
    actual = rows[0]
    # Frozen rounded output from the notebook's executed equilibrium block.
    np.testing.assert_allclose([actual["gross_sharpe"], actual["net_sharpe"], actual["spread_cost"]],
                               [2.338401, 1.306334, 0.327761], atol=5.1e-7, rtol=0)
    audit["notebook_saved_baseline_reproduced"] = True
    print("Baseline reproduced:", actual, flush=True)
    del strategy, accountant
    gc.collect()

    # Tighten each side toward its unchanged mid. Original quote files stay untouched.
    tight_prices = {pair: quotes.assign(
        bid=quotes.mid + FACTOR * (quotes.bid - quotes.mid),
        ask=quotes.mid + FACTOR * (quotes.ask - quotes.mid)) for pair, quotes in prices.items()}
    tight_data = prepare_basket(tight_prices, freq="1min")
    # The public strategy removes quotes outside its scheduled market week.
    idx = data.mids.index
    tight_data = replace(tight_data, mids=tight_data.mids.loc[idx], spreads=tight_data.spreads.loc[idx],
                         level=tight_data.level.loc[idx], returns=data.returns,
                         full_spread=tight_data.full_spread.loc[idx], continuous=data.continuous)
    pd.testing.assert_frame_equal(data.mids, tight_data.mids)
    np.testing.assert_allclose(tight_data.spreads, FACTOR * data.spreads, atol=1e-15, rtol=1e-10)
    accountant = PairAccountant(tight_data)
    for delay in [0, 1]:
        daily, events = save_run(accountant, targets, "tighter_fixed_trades", delay, rows)
        old_daily, old_events = baseline[delay]
        pd.testing.assert_series_equal(daily.gross, old_daily.gross)
        pd.testing.assert_frame_equal(events.drop(columns=[c for c in events if "cost" in c]),
                                      old_events.drop(columns=[c for c in old_events if "cost" in c]))
        np.testing.assert_allclose(daily.net - old_daily.net, old_daily.cost - daily.cost, atol=1e-15)
        assert ((daily.cost <= old_daily.cost + 1e-15) & (daily.cost >= 0)).all()
        audit[f"fixed_cost_ratio_delay{delay}"] = float(daily.cost.sum() / old_daily.cost.sum())
        audit[f"fixed_max_daily_error_vs_linear_cost_scaling_delay{delay}"] = float(
            (daily.cost - FACTOR * old_daily.cost).abs().max())
    print("Fixed-trade spread comparison complete. Regenerating signals with tighter quotes.", flush=True)
    del prices, data
    gc.collect()
    strategy = SGDNEEREquilibriumStrategy(list(tight_prices), tight_prices, PARAMS)
    strategy.generate_signals()
    assert frame_digest(strategy.features.drop(columns="spread")) == mid_feature_hash
    assert frame_digest(strategy.component_deviations) == component_hash
    strategy.pair_targets.to_pickle(OUTPUT / "tighter_regenerated_targets.pkl")
    strategy.entry_details.to_csv(OUTPUT / "tighter_regenerated_entries.csv.gz", compression="gzip")
    new_entries = strategy.entry_details.index
    audit.update(regenerated_entries=len(new_entries),
                 shared_entry_timestamps=len(new_entries.intersection(baseline_entries)),
                 baseline_only_entry_timestamps=len(baseline_entries.difference(new_entries)),
                 tighter_only_entry_timestamps=len(new_entries.difference(baseline_entries)),
                 mid_features_and_component_deviations_unchanged=True,
                 fixed_trade_gross_and_positions_unchanged=True)
    for delay in [0, 1]:
        daily, _ = save_run(accountant, strategy.pair_targets, "tighter_regenerated", delay, rows)
        audit[f"regenerated_trader_max_error_delay{delay}"] = verify_trader(strategy, daily, delay)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    (OUTPUT / "verification.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(comparison[comparison.period.isin(["2017-2025", "2024-2025"])].to_string(index=False), flush=True)
    print("Saved results to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
