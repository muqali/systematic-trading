"""Explain calendar variation in the frozen 24h/1.5z aligned strategy.

Uses saved signals and fills; no parameter fitting or new trade selection.
Run from any working directory with the project's Python environment.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE
from research.sgd_neer_codex.sgd_neer_multiday import daily_bootstrap
from research.sgd_neer_codex.sgd_neer_timing import rolling_marks, rolling_features

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sgd_neer_results/subset"
OUTPUT = HERE / "sgd_neer_results/regime_diagnostics"
TRIGGER = "equilibrium_24h_z1.5"
MAIN = f"{TRIGGER}__aligned__risk_matched"
PERIODS = {"2016-2018": (2016, 2018), "2019-2021": (2019, 2021),
           "2022-2025": (2022, 2025)}


def timestamps(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_convert("America/New_York")


def periods(years):
    return np.select([(years >= start) & (years <= end)
                      for start, end in PERIODS.values()], list(PERIODS), default="outside")


def sharpe(values):
    return np.sqrt(252) * values.mean() / values.std() if values.std() > 0 else np.nan


def pnl_summary(frame):
    return dict(days=len(frame), gross_bp=frame.gross.sum()*1e4,
                cost_bp=frame.cost.sum()*1e4, net_bp=frame.net.sum()*1e4,
                gross_sharpe=sharpe(frame.gross), net_sharpe=sharpe(frame.net),
                annual_net_vol_pct=frame.net.std()*np.sqrt(252)*100,
                cost_per_turnover_bp=frame.cost.sum()/frame.turnover.sum()*1e4)


def episode_summary(frame):
    wins, losses = frame.loc[frame.net > 0, "net"], frame.loc[frame.net < 0, "net"]
    normal = frame[~frame.timeout]
    return dict(trades=len(frame), win_pct=(frame.net > 0).mean()*100,
                gross_bp=frame.gross.sum()*1e4, cost_bp=frame.cost.sum()*1e4,
                net_bp=frame.net.sum()*1e4, mean_win_bp=wins.mean()*1e4,
                mean_loss_bp=losses.mean()*1e4, timeout_pct=frame.timeout.mean()*100,
                timeout_net_bp=frame.loc[frame.timeout, "net"].sum()*1e4,
                near_mean_net_bp=normal.net.sum()*1e4,
                mean_hours=frame.market_hours.mean(),
                mean_entry_deviation_bp=frame.entry_deviation.abs().mean()*1e4,
                mean_entry_z=frame.entry_z.abs().mean(),
                mean_exit_index_fade_bp=frame.index_fade.mean()*1e4,
                mean_exit_subset_fade_bp=frame.subset_fade.mean()*1e4,
                mean_exit_unit_cost_bp=(frame.cost/frame.actual_units.abs()).mean()*1e4,
                mean_entry_index_vol_bp=frame.entry_index_vol.mean()*1e4,
                mean_selected=frame.selected_count.mean(),
                mean_units=frame.actual_units.abs().mean(),
                mean_entry_spread_bp=frame.entry_spread.mean()*1e4,
                mean_exit_spread_bp=frame.exit_spread.mean()*1e4,
                near_mean_price_fade_bp=normal.index_fade.mean()*1e4,
                near_mean_reference_catchup_bp=normal.reference_catchup.mean()*1e4,
                near_mean_index_loss_pct=(normal.index_fade < 0).mean()*100)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    yearly, grouped, daily = [], [], {}
    for variant in ["aligned__risk_matched", "full__retained", "aligned__retained", "aligned__normalized"]:
        for delay in [0, 1]:
            frame = pd.read_csv(SOURCE/f"daily_{TRIGGER}__{variant}_fill{delay}.csv.gz", index_col=0, parse_dates=True)
            daily[variant, delay] = frame
            for year, part in frame.groupby(frame.index.year):
                yearly.append(dict(variant=variant, delay=delay, year=year, **pnl_summary(part)))
            for name, (start, end) in PERIODS.items():
                part = frame.loc[str(start):str(end)]
                grouped.append(dict(variant=variant, delay=delay, period=name, **pnl_summary(part)))
    annual, period_pnl = pd.DataFrame(yearly), pd.DataFrame(grouped)
    annual.to_csv(OUTPUT/"annual_pnl.csv", index=False)
    period_pnl.to_csv(OUTPUT/"period_pnl.csv", index=False)
    main_daily = daily["aligned__risk_matched", 0]
    main_daily.resample("QE").sum().to_csv(OUTPUT/"quarterly_pnl.csv")
    intervals = [dict(period=name, **daily_bootstrap(main_daily.loc[str(start):str(end), "net"]))
                 for name, (start, end) in PERIODS.items()]
    pd.DataFrame(intervals).to_csv(OUTPUT/"sharpe_intervals.csv", index=False)

    print("Loading the existing minute basket", flush=True)
    data = pd.read_pickle(CACHE/"data.pkl")
    marks = rolling_marks(data)
    features = rolling_features(marks, "equilibrium", 24)
    deviation = -2*features.edge
    reference = marks.level-deviation
    decisions = pd.read_csv(SOURCE/f"decisions_{MAIN}.csv.gz")
    ledger = pd.read_csv(SOURCE/f"trades_{MAIN}_fill0.csv.gz")
    entries, exits = ledger.iloc[::2].reset_index(drop=True), ledger.iloc[1::2].reset_index(drop=True)
    assert len(entries) == len(exits) == len(decisions)
    assert (entries.before == 0).all() and (exits.after == 0).all()
    entry_times, exit_times = timestamps(entries.timestamp), timestamps(exits.timestamp)
    assert entry_times.equals(timestamps(decisions.timestamp))
    a, b = data.mids.index.get_indexer(entry_times), data.mids.index.get_indexer(exit_times)
    assert min(a.min(), b.min()) >= 0
    pairs = list(data.mids.columns)
    positions = entries[[f"after_{p}" for p in pairs]].to_numpy()
    log_change = np.log(data.mids.iloc[b].to_numpy()/data.mids.iloc[a].to_numpy())
    leg_gross = positions*log_change
    leg_cost = (entries[[f"cost_{p}" for p in pairs]].to_numpy()
                + exits[[f"cost_{p}" for p in pairs]].to_numpy())
    episodes = decisions.copy()
    episodes["timestamp"], episodes["exit_timestamp"] = entry_times, exit_times
    episodes["year"], episodes["period"] = entry_times.year, periods(entry_times.year)
    episodes["direction"] = np.where(episodes.actual_units > 0, "long_SGD", "short_SGD")
    episodes["gross"] = leg_gross.sum(axis=1)
    episodes["cost"] = leg_cost.sum(axis=1)
    episodes["net"] = episodes.gross-episodes.cost
    np.testing.assert_allclose(episodes[["gross", "cost", "net"]].sum(),
                               main_daily[["gross", "cost", "net"]].sum(), atol=1e-10)
    signed = np.sign(episodes.actual_units.to_numpy())
    episodes["market_hours"] = (marks.index.get_indexer(exit_times)-marks.index.get_indexer(entry_times))/60
    episodes["timeout"] = episodes.market_hours >= 48
    episodes["entry_deviation"] = deviation.reindex(entry_times).to_numpy()
    np.testing.assert_allclose(episodes.entry_deviation, episodes.index_deviation, atol=1e-12)
    episodes["entry_z"] = features.z.reindex(entry_times).to_numpy()
    episodes["exit_z"] = features.z.reindex(exit_times).to_numpy()
    episodes["entry_index_vol"] = marks.vol.reindex(entry_times).to_numpy()
    episodes["entry_spread"] = data.full_spread.reindex(entry_times).to_numpy()
    episodes["exit_spread"] = data.full_spread.reindex(exit_times).to_numpy()
    episodes["index_fade"] = signed*(data.level.iloc[b].to_numpy()-data.level.iloc[a].to_numpy())
    episodes["subset_fade"] = episodes.gross/episodes.actual_units.abs()
    episodes["reference_catchup"] = -signed*(reference.reindex(exit_times).to_numpy()-reference.reindex(entry_times).to_numpy())
    for j, pair in enumerate(pairs):
        episodes[f"gross_{pair}"] = leg_gross[:, j]
        episodes[f"cost_{pair}"] = leg_cost[:, j]
        episodes[f"net_{pair}"] = leg_gross[:, j]-leg_cost[:, j]
    episodes.to_csv(OUTPUT/"episodes.csv", index=False)
    for key in ["year", "period", "direction"]:
        pd.DataFrame([dict(**{key:name}, **episode_summary(part))
                      for name, part in episodes.groupby(key)]).to_csv(OUTPUT/f"episodes_by_{key}.csv", index=False)
    attribution = []
    for period, part in episodes.groupby("period"):
        for pair in pairs:
            attribution.append(dict(period=period, pair=pair,
                                    gross_bp=part[f"gross_{pair}"].sum()*1e4,
                                    cost_bp=part[f"cost_{pair}"].sum()*1e4,
                                    net_bp=part[f"net_{pair}"].sum()*1e4))
    pd.DataFrame(attribution).to_csv(OUTPUT/"pair_attribution.csv", index=False)
    direction = [dict(period=period, direction=direction, **episode_summary(part))
                 for (period, direction), part in episodes.groupby(["period", "direction"])]
    pd.DataFrame(direction).to_csv(OUTPUT/"direction_by_period.csv", index=False)

    # Fixed-horizon, unscaled index markouts from the same frozen entry times.
    # Diagnostic only: observations can overlap and positions may have exited.
    markout = []
    start = marks.index.get_indexer(entry_times)
    level = marks.level.to_numpy()
    for hours in [1, 6, 24, 48]:
        end = start + hours*60
        valid = end < len(marks)
        moves = np.full(len(start), np.nan)
        moves[valid] = signed[valid]*(level[end[valid]]-level[start[valid]])
        for period in PERIODS:
            sample = moves[episodes.period.eq(period)]
            sample = sample[np.isfinite(sample)]
            markout.append(dict(period=period, horizon_hours=hours, trades=len(sample),
                                mean_fade_bp=sample.mean()*1e4, median_fade_bp=np.median(sample)*1e4,
                                win_pct=(sample > 0).mean()*100))
    pd.DataFrame(markout).to_csv(OUTPUT/"index_markouts.csv", index=False)

    components = pd.read_csv(SOURCE/"component_outcomes.csv.gz")
    components = components[components.trigger.eq(TRIGGER)].copy()
    components["period"] = periods(timestamps(components.timestamp).year)
    component_rows = []
    for (period, aligned), part in components.groupby(["period", "aligned"]):
        component_rows.append(dict(period=period, aligned=aligned, cross_observations=len(part),
                                   mean_gross_bp=part.gross_fade_return.mean()*1e4,
                                   mean_cost_bp=part.cost.mean()*1e4,
                                   mean_net_bp=part.net_fade_return.mean()*1e4))
    pd.DataFrame(component_rows).to_csv(OUTPUT/"component_outcomes_by_period.csv", index=False)
    # Synthetic SGD-cross attribution includes each cross's share of the
    # USDSGD hedge, rather than treating the quoted JPY leg as the whole trade.
    components["timestamp"] = timestamps(components.timestamp)
    exposure_columns = [f"exposure_{ccy}" for ccy in ["USD", "EUR", "AUD", "GBP", "JPY"]]
    components = components.merge(episodes[["timestamp", *exposure_columns]],
                                  on="timestamp", validate="many_to_one")
    components["exposure"] = [row[f"exposure_{row.currency}"] for _, row in components.iterrows()]
    components["actual_gross"] = components.exposure.abs()*components.gross_fade_return
    components["actual_cost"] = components.exposure.abs()*components.cost
    components["actual_net"] = components.actual_gross-components.actual_cost
    np.testing.assert_allclose(components[["actual_gross", "actual_cost", "actual_net"]].sum(),
                               episodes[["gross", "cost", "net"]].sum(), atol=1e-10)
    cross_attribution = components.groupby(["period", "currency"]).agg(
        gross_bp=("actual_gross", lambda s: s.sum()*1e4),
        cost_bp=("actual_cost", lambda s: s.sum()*1e4),
        net_bp=("actual_net", lambda s: s.sum()*1e4),
        selected_entries=("exposure", lambda s: int((s != 0).sum())))
    cross_attribution.to_csv(OUTPUT/"synthetic_cross_attribution.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), layout="constrained")
    fig.suptitle("24h equilibrium / 1.5z: what changed in 2019–2021?", fontsize=16)
    for variant, label, color in [("aligned__risk_matched", "Aligned, risk matched", "#116b89"),
                                  ("full__retained", "Full-index control", "#8997a5")]:
        series = daily[variant, 0].net.cumsum()*100
        axes[0].plot(series.index, series, label=label, color=color)
    axes[0].axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-01"), color="#dba869", alpha=.22)
    axes[0].set(title="Same-quote cumulative net PnL", ylabel="PnL per unit capital (%)")
    axes[0].legend(loc="upper left", frameon=False)
    main_annual = annual[annual.variant.eq("aligned__risk_matched")]
    for delay, shift, color, label in [(0, -.18, "#116b89", "Same quote"),
                                       (1, .18, "#94b9c7", "Next common quote")]:
        part = main_annual[main_annual.delay.eq(delay)]
        axes[1].bar(part.year+shift, part.net_sharpe, width=.36, color=color, label=label)
    axes[1].set(title="The weak years were 2019–2021; 2022 had already recovered",
                ylabel="Annualised net Sharpe", xticks=range(2016, 2026))
    axes[1].axhline(0, color="#4b5563", linewidth=.8)
    axes[1].legend(frameon=False)
    markout_frame = pd.DataFrame(markout)
    for period, color in zip(PERIODS, ["#116b89", "#bd6337", "#568549"]):
        part = markout_frame[markout_frame.period.eq(period)]
        axes[2].plot(part.horizon_hours, part.mean_fade_bp, marker="o", color=color, label=period)
    axes[2].set(title="Actual index reversals following the frozen entry signals",
                xlabel="Scheduled market hours after entry", ylabel="Mean gross fade return (bp)",
                xticks=[1, 6, 24, 48])
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Recorded spreads included in PnL; financing excluded. Markouts are unscaled diagnostics, before costs.", fontsize=9)
    fig.savefig(OUTPUT/"regime_diagnostics.png", dpi=160)
    plt.close(fig)

    metadata = dict(policy=MAIN, execution_delay=0, input_cache=str(CACHE/"data.pkl"),
                    financing="excluded", periods=PERIODS,
                    annual_pnl="marked to market on NY trading dates; fixed log-return notional units",
                    trade_grouping="whole episodes assigned to entry year; totals need not match annual marked-to-market PnL",
                    diagnostics="retrospective attribution; no parameters fitted or signals changed",
                    bootstrap="2000 circular resamples of 20 trading days; conditional on the already examined strategy and periods",
                    exit_type="48 or more scheduled market hours classified as timeout; others near/cross mean",
                    verified_total_pnl=True, minute_rows=len(data.mids), episodes=len(episodes))
    (OUTPUT/"method.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("ANNUAL\n", annual.query("variant == 'aligned__risk_matched' and delay == 0").round(3).to_string(index=False))
    print("EPISODES\n", pd.read_csv(OUTPUT/"episodes_by_period.csv").round(3).to_string(index=False))
    print("YEARLY TRADES\n", pd.read_csv(OUTPUT/"episodes_by_year.csv").round(3).to_string(index=False))
    print("FIXED HORIZON\n", pd.DataFrame(markout).round(3).to_string(index=False))
    print("PAIR ATTRIBUTION\n", pd.DataFrame(attribution).round(3).to_string(index=False))
    print("COMPONENTS\n",pd.DataFrame(component_rows).round(3).to_string(index=False))
    print("Saved diagnostics to", OUTPUT, flush=True)


if __name__ == "__main__":
    run()
