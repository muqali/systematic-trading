"""Entry-calendar attribution for the frozen notebook 24h/1.5z strategy.

Uses original-spread trades from spread_sensitivity, with indicators starting
in 2017. This is descriptive research: no signals, filters or parameters change.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sgd_neer_results/spread_sensitivity"
OUTPUT = HERE / "sgd_neer_results/calendar_diagnostics"
PERIODS = {"2017-2025": (2017, 2025), "2017-2023": (2017, 2023),
           "2024-2025": (2024, 2025), "2017-2018": (2017, 2018),
           "2019-2021": (2019, 2021), "2022-2025": (2022, 2025)}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FOUR_HOURS = [f"{h:02d}:00-{h+4:02d}:00" for h in range(0, 24, 4)]
MONTH_BINS = ["01-05", "06-10", "11-15", "16-20", "21-25", "26-end"]
DIMENSIONS = {"hour_sgt": list(range(24)), "session_sgt": FOUR_HOURS,
              "weekday_fx": WEEKDAYS[:5], "day_of_month_fx": list(range(1, 32)),
              "month_part_fx": MONTH_BINS, "month_turn_fx": ["First 3", "Middle", "Last 3"],
              "weekday_sgt": WEEKDAYS, "month_part_sgt": MONTH_BINS,
              "session_london": FOUR_HOURS, "session_ny": FOUR_HOURS}
PRIMARY = ["session_sgt", "weekday_fx", "month_part_fx"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamps(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def reconstruct(data, delay):
    ledger = pd.read_csv(SOURCE / f"baseline_delay{delay}_events.csv.gz")
    decisions = pd.read_csv(SOURCE / "baseline_entries.csv.gz")
    entries, exits = ledger.iloc[::2].reset_index(drop=True), ledger.iloc[1::2].reset_index(drop=True)
    assert len(entries) == len(exits) == len(decisions) == 804
    assert (entries.before == 0).all() and (exits.after == 0).all()
    assert (entries.after > 0).all() and (exits.before > 0).all()
    pairs = list(data.mids)
    positions = entries[[f"after_{p}" for p in pairs]].to_numpy()
    np.testing.assert_allclose(positions, exits[[f"before_{p}" for p in pairs]], atol=1e-14)
    entry, exit = stamps(entries.timestamp), stamps(exits.timestamp)
    requested = stamps(decisions.timestamp)
    a, b = data.mids.index.get_indexer(entry), data.mids.index.get_indexer(exit)
    assert (a >= 0).all() and (b > a).all()
    assert (entry >= requested).all()
    if delay == 0:
        assert entry.equals(requested)
    gross = positions * (np.log(data.mids.iloc[b].to_numpy()) - np.log(data.mids.iloc[a].to_numpy()))
    costs = entries[[f"cost_{p}" for p in pairs]].to_numpy() + exits[[f"cost_{p}" for p in pairs]].to_numpy()
    frame = decisions.copy()
    frame["signal_timestamp"], frame["entry_timestamp"], frame["exit_timestamp"] = requested, entry, exit
    frame["gross"], frame["cost"] = gross.sum(axis=1), costs.sum(axis=1)
    frame["net"] = frame.gross - frame.cost
    frame["units"] = frame.actual_units.abs()
    frame["unit_net_bp"] = frame.net / frame.units * 1e4
    frame["net_bp"] = frame.net * 1e4
    frame["direction"] = np.where(frame.actual_units > 0, "Long SGD", "Short SGD")
    frame["wall_holding_hours"] = (exit - entry) / pd.Timedelta(hours=1)
    for j, pair in enumerate(pairs):
        frame[f"gross_{pair}"] = gross[:, j]
        frame[f"cost_{pair}"] = costs[:, j]
    ny = requested.tz_convert("America/New_York")
    sg = requested.tz_convert("Asia/Singapore")
    london = requested.tz_convert("Europe/London")
    fx_date = (ny.tz_localize(None) + pd.Timedelta(hours=7)).normalize()
    assert (fx_date.dayofweek < 5).all()
    frame["year"], frame["entry_month"] = fx_date.year, fx_date.to_period("M").astype(str)
    frame["fx_date"], frame["sgt_timestamp"] = fx_date, sg
    frame["hour_sgt"] = sg.hour
    for name, values in [("sgt", sg), ("ny", ny), ("london", london)]:
        frame[f"session_{name}"] = np.asarray(FOUR_HOURS)[values.hour // 4]
    frame["weekday_fx"] = np.asarray(WEEKDAYS)[fx_date.dayofweek]
    frame["weekday_sgt"] = np.asarray(WEEKDAYS)[sg.dayofweek]
    frame["day_of_month_fx"] = fx_date.day
    frame["month_part_fx"] = np.asarray(MONTH_BINS)[np.minimum((fx_date.day-1)//5, 5)]
    frame["month_part_sgt"] = np.asarray(MONTH_BINS)[np.minimum((sg.day-1)//5, 5)]
    dates = fx_date.to_numpy().astype("datetime64[D]")
    starts = dates.astype("datetime64[M]").astype("datetime64[D]")
    ends = (dates.astype("datetime64[M]") + 1).astype("datetime64[D]")
    first, last = np.busday_count(starts, dates) + 1, np.busday_count(dates, ends)
    frame["month_turn_fx"] = np.select([first <= 3, last <= 3], ["First 3", "Last 3"], default="Middle")
    daily = pd.read_csv(SOURCE / f"baseline_delay{delay}_daily.csv.gz", index_col=0, parse_dates=True)
    error = float((frame[["gross", "cost", "net"]].sum() - daily[["gross", "cost", "net"]].sum()).abs().max())
    np.testing.assert_allclose(frame[["gross", "cost", "net"]].sum(),
                               daily[["gross", "cost", "net"]].sum(), atol=1e-10, rtol=0)
    return frame, error


def summary(part):
    n = len(part)
    pnl = part.net if n else pd.Series(dtype=float)
    trimmed = pnl.sort_values().iloc[int(.05*n):n-int(.05*n)] if n else pnl
    winners = pnl[pnl > 0].sum()
    losers = -pnl[pnl < 0].sum()
    return dict(trades=n, mean_gross_bp=part.gross.mean()*1e4, mean_cost_bp=part.cost.mean()*1e4,
                mean_net_bp=pnl.mean()*1e4, median_net_bp=pnl.median()*1e4,
                trimmed_mean_net_bp=trimmed.mean()*1e4, win_pct=(pnl > 0).mean()*100,
                total_net_bp=pnl.sum()*1e4, profit_factor=winners/losers if losers > 0 else np.nan,
                mean_unit_net_bp=part.unit_net_bp.mean(), mean_units=part.units.mean(),
                mean_holding_wall_hours=part.wall_holding_hours.mean(),
                net_without_best_trade_bp=(pnl.sum()-pnl.max())*1e4 if n else np.nan)


def bootstrap_means(frame, dimension, repetitions=4000):
    """Sample consecutive three-calendar-month blocks; preserve whole trades.

    Returns pointwise intervals, not simultaneous confidence intervals. Results
    condition on the already examined strategy; no claim of independent holdout.
    """
    month_grid = pd.period_range("2017-01", "2025-12", freq="M").astype(str)
    labels = DIMENSIONS[dimension]
    sums = frame.pivot_table(index="entry_month", columns=dimension, values="net_bp", aggfunc="sum")
    counts = frame.pivot_table(index="entry_month", columns=dimension, values="net_bp", aggfunc="size")
    sums = sums.reindex(index=month_grid, columns=labels).fillna(0).to_numpy()
    counts = counts.reindex(index=month_grid, columns=labels).fillna(0).to_numpy()
    rng = np.random.default_rng(20260928)
    n_months, block = len(month_grid), 3
    starts = rng.integers(0, n_months, size=(repetitions, n_months//block))
    indices = ((starts[:, :, None] + np.arange(block)) % n_months).reshape(repetitions, n_months)
    weights = np.stack([np.bincount(row, minlength=n_months) for row in indices])
    bs_sum, bs_count = weights @ sums, weights @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        means = bs_sum / bs_count
        others = (bs_sum.sum(axis=1, keepdims=True)-bs_sum)/(bs_count.sum(axis=1, keepdims=True)-bs_count)
    rows = []
    for j, label in enumerate(labels):
        if counts[:, j].sum() == 0:
            continue
        low, high = np.nanpercentile(means[:, j], [2.5, 97.5])
        dlow, dhigh = np.nanpercentile(means[:, j]-others[:, j], [2.5, 97.5])
        rows.append(dict(dimension=dimension, bucket=label, mean_ci_low_bp=low, mean_ci_high_bp=high,
                         difference_vs_rest_low_bp=dlow, difference_vs_rest_high_bp=dhigh))
    return rows


def adjusted_tests(frame, delay):
    formula = "net_bp ~ C(year) + C(direction) + units + " + " + ".join(f"C({d})" for d in PRIMARY)
    model = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame.entry_month})
    rows = []
    names = list(model.params.index)
    for dimension in PRIMARY:
        indices = [j for j, name in enumerate(names) if name.startswith(f"C({dimension})[")]
        restriction = np.eye(len(names))[indices]
        test = model.f_test(restriction)
        rows.append(dict(delay=delay, dimension=dimension, f_stat=float(test.fvalue),
                         p_value=float(test.pvalue), df_num=float(test.df_num), df_denom=float(test.df_denom)))
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = multipletests(result.p_value, method="holm")[1]
    (OUTPUT / f"adjusted_model_delay{delay}.txt").write_text(model.summary().as_text()+"\n")
    return result


def plot(summary_table, intervals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    fig.suptitle("24h equilibrium / 1.5z: PnL by entry calendar", fontsize=17)
    labels = ["Entry time (Singapore)", "Entry FX trading weekday", "Entry day within FX trading month"]
    for i, (dimension, title) in enumerate(zip(PRIMARY, labels)):
        full = summary_table.query("delay == 0 and period == '2017-2025' and direction == 'All' and dimension == @dimension").set_index("bucket").reindex(DIMENSIONS[dimension])
        ci = intervals.query("dimension == @dimension").set_index("bucket").reindex(DIMENSIONS[dimension])
        x = np.arange(len(full))
        ax = axes[i, 0]
        for shift, column, color, label in [(-.2, "mean_gross_bp", "#a4c4cd", "Gross"),
                                           (.2, "mean_net_bp", "#186b83", "Net")]:
            ax.bar(x+shift, full[column], width=.38, color=color, label=label)
        ax.errorbar(x+.2, full.mean_net_bp,
                    yerr=[np.maximum(0, full.mean_net_bp-ci.mean_ci_low_bp),
                          np.maximum(0, ci.mean_ci_high_bp-full.mean_net_bp)],
                    fmt="none", color="#333333", capsize=3, linewidth=1)
        ax.set_xticks(x, [f"{v}\nn={int(n)}" for v, n in zip(full.index, full.trades)], fontsize=8)
        ax.set(title=title, ylabel="Mean whole-trade PnL (bp)")
        if i == 0:
            ax.legend(frameon=False, ncol=2)
        ax = axes[i, 1]
        for period, color, marker in [("2017-2023", "#8c8c8c", "o"), ("2024-2025", "#c77336", "s")]:
            part = summary_table.query("delay == 0 and period == @period and direction == 'All' and dimension == @dimension").set_index("bucket").reindex(DIMENSIONS[dimension])
            ax.plot(x, part.mean_net_bp, marker=marker, color=color, label=period)
        ax.set_xticks(x, full.index, fontsize=8)
        ax.set(title="Consistency across periods", ylabel="Mean net PnL per trade (bp)")
        if i == 0:
            ax.legend(frameon=False)
    for ax in axes.flat:
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Original spreads; same-quote fills; financing excluded. Error bars: pointwise 95% three-month-block bootstrap.\nWeekday/month date roll at 17:00 New York. Complete PnL assigned to entry, including subsequent days.", fontsize=9)
    fig.savefig(OUTPUT / "entry_calendar.png", dpi=160)
    fig.savefig(OUTPUT / "entry_calendar.pdf")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = dict(strategy="24h / 1.5z / aligned / risk_matched / 48h", original_spreads=True,
                 history="2017-2025, indicators initialized in 2017", financing=False,
                 grouping="Full completed-trade PnL assigned to entry signal calendar, unchanged for delay sensitivity",
                 hours="Asia/Singapore; London and New York cross-checks",
                 dates="FX trading date: New York local clock plus 7 hours, then normalize; Singapore calendar also supplied",
                 month_turn="First/last three Mon-Fri business dates; no holiday-calendar adjustment",
                 primary_dimensions=PRIMARY, primary_buckets={d: DIMENSIONS[d] for d in PRIMARY},
                 uncertainty="4000 circular bootstrap resamples of three consecutive calendar-month blocks",
                 adjusted_tests="Joint calendar categories; control year, SGD direction, units and other primary calendar dimensions; cluster by entry month; Holm across three tests",
                 changes_to_strategy=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(audit, indent=2)+"\n")
    print("Loading cached quotes and reconstructing the saved 804 completed trades.", flush=True)
    data = pd.read_pickle(CACHE / "data.pkl")
    tables, annual, intervals, tests, daily_tables = [], [], [], [], []
    for delay in [0, 1]:
        frame, error = reconstruct(data, delay)
        frame.to_csv(OUTPUT / f"episodes_delay{delay}.csv.gz", index=False, compression="gzip")
        audit[f"total_pnl_reconciliation_error_delay{delay}"] = error
        for period, (start, end) in PERIODS.items():
            sample = frame[frame.year.between(start, end)]
            for dimension, buckets in DIMENSIONS.items():
                for bucket in buckets:
                    subset = sample[sample[dimension] == bucket]
                    for direction in (["All", "Long SGD", "Short SGD"] if dimension in PRIMARY else ["All"]):
                        selected = subset if direction == "All" else subset[subset.direction == direction]
                        tables.append(dict(delay=delay, period=period, dimension=dimension, bucket=bucket,
                                           direction=direction, **summary(selected)))
        for year, sample in frame.groupby("year"):
            for dimension in PRIMARY:
                for bucket in DIMENSIONS[dimension]:
                    annual.append(dict(delay=delay, year=year, dimension=dimension, bucket=bucket,
                                       **summary(sample[sample[dimension] == bucket])))
        tests.append(adjusted_tests(frame, delay))
        if delay == 0:
            for dimension in PRIMARY + ["month_turn_fx"]:
                intervals.extend(bootstrap_means(frame, dimension))
        daily = pd.read_csv(SOURCE / f"baseline_delay{delay}_daily.csv.gz", index_col=0, parse_dates=True)
        daily["weekday_fx"] = np.asarray(WEEKDAYS)[daily.index.dayofweek]
        daily["month_part_fx"] = np.asarray(MONTH_BINS)[np.minimum((daily.index.day-1)//5, 5)]
        for dimension in ["weekday_fx", "month_part_fx"]:
            for bucket, sample in daily.groupby(dimension):
                daily_tables.append(dict(delay=delay, dimension=dimension, bucket=bucket, days=len(sample),
                                         mean_gross_bp=sample.gross.mean()*1e4, mean_cost_bp=sample.cost.mean()*1e4,
                                         mean_net_bp=sample.net.mean()*1e4, total_net_bp=sample.net.sum()*1e4))
        print(f"Delay {delay}: totals reconcile; calendar summaries and adjusted tests complete.", flush=True)
    table, ci = pd.DataFrame(tables), pd.DataFrame(intervals)
    table.to_csv(OUTPUT / "entry_summary.csv", index=False)
    pd.DataFrame(annual).to_csv(OUTPUT / "entry_summary_annual.csv", index=False)
    ci.to_csv(OUTPUT / "bootstrap_intervals.csv", index=False)
    pd.concat(tests).to_csv(OUTPUT / "adjusted_tests.csv", index=False)
    pd.DataFrame(daily_tables).to_csv(OUTPUT / "daily_accrual_summary.csv", index=False)
    audit["input_hashes"] = {str(path.relative_to(HERE.parents[3])): sha256(path) for path in [
        CACHE / "data.pkl", SOURCE / "baseline_entries.csv.gz", SOURCE / "baseline_delay0_events.csv.gz",
        SOURCE / "baseline_delay1_events.csv.gz", SOURCE / "baseline_delay0_daily.csv.gz",
        SOURCE / "baseline_delay1_daily.csv.gz", SOURCE / "verification.json"]}
    audit["script_sha256"] = sha256(Path(__file__).resolve())
    (OUTPUT / "verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    plot(table, ci)
    columns = ["dimension", "bucket", "trades", "mean_gross_bp", "mean_cost_bp", "mean_net_bp", "median_net_bp", "win_pct", "mean_unit_net_bp"]
    print(table.query("delay == 0 and period == '2017-2025' and direction == 'All' and dimension in @PRIMARY")[columns].round(3).to_string(index=False))
    print(pd.concat(tests).to_string(index=False))
    print("Saved diagnostics to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
