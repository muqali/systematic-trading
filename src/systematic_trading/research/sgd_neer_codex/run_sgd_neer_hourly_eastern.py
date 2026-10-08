"""Hourly US/Eastern entry-PnL attribution for the frozen notebook strategy."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.run_sgd_neer_calendar_diagnostics import PERIODS, sha256, summary

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sgd_neer_results/calendar_diagnostics"
OUTPUT = HERE / "sgd_neer_results/hourly_eastern"


def bootstrap(frame):
    months = pd.period_range("2017-01", "2025-12", freq="M").astype(str)
    sums = frame.pivot_table(index="entry_month", columns="hour_eastern", values="net_bp", aggfunc="sum")
    counts = frame.pivot_table(index="entry_month", columns="hour_eastern", values="net_bp", aggfunc="size")
    sums = sums.reindex(index=months, columns=range(24)).fillna(0).to_numpy()
    counts = counts.reindex(index=months, columns=range(24)).fillna(0).to_numpy()
    rng = np.random.default_rng(20260928)
    starts = rng.integers(0, len(months), size=(4000, len(months)//3))
    indices = ((starts[:, :, None] + np.arange(3)) % len(months)).reshape(4000, len(months))
    weights = np.stack([np.bincount(row, minlength=len(months)) for row in indices])
    sums, counts = weights @ sums, weights @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums/counts
        other = (sums.sum(axis=1, keepdims=True)-sums)/(counts.sum(axis=1, keepdims=True)-counts)
    rows = []
    for h in range(24):
        low, high = np.nanpercentile(means[:, h], [2.5, 97.5])
        dlow, dhigh = np.nanpercentile(means[:, h]-other[:, h], [2.5, 97.5])
        rows.append(dict(hour_eastern=h, mean_ci_low_bp=low, mean_ci_high_bp=high,
                         difference_vs_rest_low_bp=dlow, difference_vs_rest_high_bp=dhigh,
                         valid_bootstrap_samples=int(np.isfinite(means[:, h]).sum())))
    return pd.DataFrame(rows)


def make_plot(table, ci):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), layout="constrained", sharex=True)
    x = np.arange(24)
    base = table.query("delay == 0 and period == '2017-2025' and direction == 'All'").set_index("hour_eastern").reindex(x)
    recent = table.query("delay == 0 and period == '2024-2025' and direction == 'All'").set_index("hour_eastern").reindex(x)
    old = table.query("delay == 0 and period == '2017-2023' and direction == 'All'").set_index("hour_eastern").reindex(x)
    ci = ci.set_index("hour_eastern").reindex(x)
    fig.suptitle("24h equilibrium / 1.5z — entry hour in US/Eastern", fontsize=17)
    axes[0].bar(x-.2, base.mean_gross_bp, width=.38, color="#a4c4cd", label="Gross")
    axes[0].bar(x+.2, base.mean_net_bp, width=.38, color="#186b83", label="Net")
    axes[0].errorbar(x+.2, base.mean_net_bp,
                    yerr=[np.maximum(0, base.mean_net_bp-ci.mean_ci_low_bp),
                          np.maximum(0, ci.mean_ci_high_bp-base.mean_net_bp)],
                    fmt="none", color="#444444", capsize=2, linewidth=1)
    axes[0].set(title="2017–2025: average full-trade PnL and pointwise 95% bootstrap intervals", ylabel="PnL per trade (bp)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, old.mean_net_bp, marker="o", color="#888888", label="2017–2023")
    axes[1].plot(x, recent.mean_net_bp, marker="s", color="#c77336", label="2024–2025")
    axes[1].set(title="Net PnL across periods (see sample sizes below)", ylabel="Net PnL per trade (bp)")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].bar(x-.2, base.trades, width=.38, color="#186b83", label="2017–2025")
    axes[2].bar(x+.2, recent.trades, width=.38, color="#c77336", label="2024–2025 subset")
    axes[2].set(ylabel="Completed trades", xlabel="Entry hour, US/Eastern local clock (EST/EDT)", xticks=x,
                xticklabels=[f"{h:02d}" for h in x])
    axes[2].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelbottom=True)
    fig.supxlabel("Original spreads; same-quote execution; financing excluded. Whole-trade PnL grouped by entry signal hour.\nConfidence intervals resample three-month blocks; individual-hour comparisons are exploratory.", fontsize=9)
    fig.savefig(OUTPUT / "hourly_eastern.png", dpi=160)
    fig.savefig(OUTPUT / "hourly_eastern.pdf")
    plt.close(fig)


def make_report(table, ci, tests):
    full = table.query("delay == 0 and period == '2017-2025' and direction == 'All'").set_index("hour_eastern")
    recent = table.query("delay == 0 and period == '2024-2025' and direction == 'All'").set_index("hour_eastern")
    delayed = table.query("delay == 1 and period == '2017-2025' and direction == 'All'").set_index("hour_eastern")
    lines = ["# Hourly entry PnL in US/Eastern", "",
             "The 24h / 1.5z equilibrium strategy with aligned constituents, risk-matched sizing and a 48h holding limit is unchanged. Results use original spreads, indicators initialized in 2017 and all 804 completed trades from the notebook-matched 2017–2025 sample. Financing remains excluded.", "",
             "Timestamps are converted from UTC to `US/Eastern`, preserving EST/EDT transitions. Hour 08 means 08:00 inclusive to 09:00 exclusive. The full entry-to-exit PnL of all legs is assigned to the **entry signal hour**; this is not PnL earned within that hour. For the one-common-quote delay sensitivity, the grouping remains the original signal hour.", "",
             "Figures are basis points of the existing position-weighted portfolio PnL convention. Pooled average net PnL is +4.80 bp/trade. Recent-period counts are included because some apparently large means depend on very few observations.", "",
             "![Hourly entry results](sgd_neer_results/hourly_eastern/hourly_eastern.png)", "",
             "| Eastern entry hour | Trades | Mean gross, bp | Mean costs, bp | Mean net, bp | Win rate | 2024–25 trades | 2024–25 mean net, bp |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for h, row in full.iterrows():
        r = recent.loc[h]
        value = f"{r.mean_net_bp:+.2f}" if r.trades else "—"
        lines.append(f"| {h:02d}:00–{h+1:02d}:00 | {int(row.trades)} | {row.mean_gross_bp:.2f} | {row.mean_cost_bp:.2f} | {row.mean_net_bp:+.2f} | {row.win_pct:.1f}% | {int(r.trades)} | {value} |")
    lines += ["", "## Reading the pattern", "",
              "- **18:00–19:00** and **11:00–12:00** have the highest pooled net means, at +15.71 and +15.20 bp, from 30 and 40 trades. Both earlier- and later-period means are positive. The 11:00 hour has only three trades in 2024–2025, so its recent result is particularly imprecise.",
              "- **08:00–12:00** is a reasonably populated area of positive pooled means. Each of its four hours is also positive in 2024–2025. This describes a candidate pattern, not an evaluated entry restriction.",
              "- **00:00, 04:00, 05:00 and 19:00** have negative pooled net means; **12:00** is approximately flat. The weak hours are not all stable: 04:00 and 05:00 turn positive in 2024–2025, whereas 01:00 and 03:00 become sharply negative.",
              "- **15:00–18:00 is sparse:** the three hourly buckets contain only nine, 14 and three entries in the full sample. There are no 17:00 entries in 2024–2025. Strong-looking means there should be read with the counts.",
              "- The previous fixed Singapore windows cannot be shifted to one constant Eastern interval: Singapore has no daylight-saving change, while Eastern time does. All hourly grouping here is recomputed directly from the original timestamps.", "",
              "## Delayed execution and uncertainty", "",
              f"With one-common-quote delayed execution, pooled net means at 11:00 and 18:00 are {delayed.loc[11, 'mean_net_bp']:+.2f} and {delayed.loc[18, 'mean_net_bp']:+.2f} bp, respectively. A common-quote delay is usually one minute and can be longer across missing quotes. The detailed table includes every hour under both delays.", "",
              "An exploratory regression jointly tests the 24 hourly categories, controlling for entry year, SGD direction, position units, FX weekday and the six calendar-date bins. Standard errors cluster by entry calendar month. These are global hourly-effect tests, not significance claims for a selected hour:", "",
              "| Execution | Hourly-effect F statistic | Raw p-value |", "| --- | ---: | ---: |"]
    for row in tests:
        lines.append(f"| {'Same quote' if row['delay'] == 0 else 'Next common quote'} | {row['f_stat']:.3f} | {row['p_value']:.4f} |")
    lines += ["", "The hourly split is a follow-up to an already inspected calendar analysis. These p-values do not adjust for that broader search. Bootstrap intervals are pointwise 95% intervals from 4,000 circular resamples of three-calendar-month blocks; they do not establish that the best observed hour will remain best. Full and recent samples overlap, and neither is a new untouched holdout.", "",
              "## Reproduction and verification", "",
              "The runner reuses the previously reconciled completed-trade ledgers. It checks that every hourly grouping contains all 804 episodes and sums to the original gross, cost and net totals, and verifies US/Eastern against America/New_York hour by hour. No positions or strategy parameters are changed.", "",
              "```sh", "PYTHONDONTWRITEBYTECODE=1 poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_hourly_eastern.py", "```", "",
              "- [Hourly results, including direction and execution-delay splits](sgd_neer_results/hourly_eastern/hourly_summary.csv)",
              "- [Annual hourly results](sgd_neer_results/hourly_eastern/hourly_annual.csv)",
              "- [Bootstrap intervals](sgd_neer_results/hourly_eastern/bootstrap_intervals.csv)",
              "- [Statistical tests](sgd_neer_results/hourly_eastern/adjusted_tests.csv)",
              "- [Verification and source hashes](sgd_neer_results/hourly_eastern/verification.json)",
              "- [Chart PDF](sgd_neer_results/hourly_eastern/hourly_eastern.pdf)"]
    (HERE / "sgd_neer_hourly_eastern_report.md").write_text("\n".join(lines)+"\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, annual, tests, hashes, totals = [], [], [], {}, []
    for delay in [0, 1]:
        path = SOURCE / f"episodes_delay{delay}.csv.gz"
        frame = pd.read_csv(path)
        hashes[str(path.relative_to(HERE))] = sha256(path)
        timestamp = pd.DatetimeIndex(pd.to_datetime(frame.signal_timestamp, utc=True))
        eastern = timestamp.tz_convert("US/Eastern")
        assert np.array_equal(eastern.hour, timestamp.tz_convert("America/New_York").hour)
        frame["hour_eastern"] = eastern.hour
        frame["timestamp_eastern"] = eastern
        assert len(frame) == 804
        assert frame.groupby("hour_eastern").size().sum() == 804
        np.testing.assert_allclose(frame.groupby("hour_eastern")[["gross", "cost", "net"]].sum().sum(),
                                   frame[["gross", "cost", "net"]].sum(), atol=1e-12, rtol=0)
        totals.append(dict(delay=delay, trades=len(frame), **frame[["gross", "cost", "net"]].sum().to_dict()))
        for period, (start, end) in PERIODS.items():
            for direction in ["All", "Long SGD", "Short SGD"]:
                sample = frame[frame.year.between(start, end)]
                if direction != "All":
                    sample = sample[sample.direction == direction]
                for hour in range(24):
                    rows.append(dict(delay=delay, period=period, direction=direction, hour_eastern=hour,
                                     **summary(sample[sample.hour_eastern == hour])))
        for year, sample in frame.groupby("year"):
            for hour in range(24):
                annual.append(dict(delay=delay, year=year, hour_eastern=hour,
                                   **summary(sample[sample.hour_eastern == hour])))
        formula = "net_bp ~ C(year) + C(direction) + units + C(weekday_fx) + C(month_part_fx) + C(hour_eastern)"
        model = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame.entry_month})
        indices = [i for i, name in enumerate(model.params.index) if name.startswith("C(hour_eastern)[")]
        test = model.f_test(np.eye(len(model.params))[indices])
        tests.append(dict(delay=delay, f_stat=float(test.fvalue), p_value=float(test.pvalue),
                          df_num=float(test.df_num), df_denom=float(test.df_denom)))
        (OUTPUT / f"adjusted_model_delay{delay}.txt").write_text(model.summary().as_text()+"\n")
        if delay == 0:
            ci = bootstrap(frame)
        frame.to_csv(OUTPUT / f"episodes_eastern_delay{delay}.csv.gz", index=False, compression="gzip")
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "hourly_summary.csv", index=False)
    pd.DataFrame(annual).to_csv(OUTPUT / "hourly_annual.csv", index=False)
    pd.DataFrame(tests).to_csv(OUTPUT / "adjusted_tests.csv", index=False)
    ci.to_csv(OUTPUT / "bootstrap_intervals.csv", index=False)
    audit = dict(timezone="US/Eastern (EST/EDT)", grouping="original entry signal hour; whole-trade PnL",
                 original_spreads=True, financing=False, totals=totals,
                 input_hashes=hashes, script_sha256=sha256(Path(__file__).resolve()),
                 helper_sha256=sha256(HERE / "run_sgd_neer_calendar_diagnostics.py"),
                 all_hourly_pnl_reconciled=True, timezone_alias_verified=True)
    (OUTPUT / "verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    make_plot(table, ci)
    make_report(table, ci, tests)
    print(table.query("delay == 0 and period == '2017-2025' and direction == 'All'")[[
        "hour_eastern", "trades", "mean_gross_bp", "mean_cost_bp", "mean_net_bp", "win_pct"]].round(3).to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print("Saved results to", OUTPUT)


if __name__ == "__main__":
    main()
