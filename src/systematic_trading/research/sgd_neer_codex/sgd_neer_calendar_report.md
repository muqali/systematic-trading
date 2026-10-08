# Trade PnL by entry time, weekday and date within the month

The strongest descriptive pattern is **entry time of day**. Entries at
20:00–24:00 Singapore time earned positive net PnL in each of the nine entry
years. The 04:00–08:00 window has the highest average profit, but only 54 trades.
Weekday and day-of-month differences have much weaker statistical support.
These results describe the existing strategy; no entry restrictions have been
added and no filtered strategy has been backtested.

This analysis uses the notebook-matched **24h equilibrium / 1.5z / aligned /
risk_matched / 48h** strategy, one-minute quotes, **original spreads**, same-quote
execution and indicators initialized in 2017. It reconstructs all **804 completed
trades** from 2017–2025. Their total gross PnL, costs and net PnL reconcile to the
saved notebook baseline: 71.358%, 32.776% and 38.582%, respectively. Average trade
PnL is **8.88 bp gross, 4.08 bp cost and 4.80 bp net**.

![Entry calendar diagnostics](sgd_neer_results/calendar_diagnostics/entry_calendar.png)

## Interpretation and calendar conventions

Each observation is one complete portfolio episode, combining all traded legs,
from entry to exit. Its full PnL, including both entry and exit costs, is assigned
to the **entry signal's timestamp**, even when it exits on another day. Entries
are disjoint because the strategy holds only one index position at a time.

Hours below use **Asia/Singapore (UTC+8)**. Weekdays and dates within the month
use the **FX trading date**, obtained by adding seven hours to the New York local
clock and taking the resulting date. Thus the date rolls at 17:00 New York;
Sunday-evening New York entries belong to Monday. London, New York and Singapore
calendar cross-checks are also saved.

All bp figures refer to the existing position-weighted portfolio log-return PnL
convention, not an unscaled currency return. Results normalized by aggregate
SGD-cross units are saved as a sizing sensitivity. Financing remains excluded.

## Time of day

| Entry time, Singapore | Trades | Mean gross, bp | Mean costs, bp | Mean net, bp | Win rate | Mean net in 2024–2025, bp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00:00–04:00 | 81 | 8.59 | 3.61 | 4.98 | 72.8% | 12.15 |
| 04:00–08:00 | 54 | 16.54 | 4.66 | **11.88** | 79.6% | **18.56** |
| 08:00–12:00 | 158 | 7.63 | 3.77 | 3.86 | 64.6% | −3.14 |
| 12:00–16:00 | 156 | 7.17 | 4.83 | 2.35 | 65.4% | **−9.68** |
| 16:00–20:00 | 165 | 5.84 | 3.70 | 2.14 | 69.1% | 8.55 |
| 20:00–24:00 | 190 | 11.90 | 4.08 | **7.82** | 73.2% | **6.17** |

- **20:00–24:00 is the most consistent window:** 190 trades, positive total net
  PnL in all nine entry years, +8.20 bp/trade in 2017–2023 and +6.17 bp in
  2024–2025. It also averaged +6.14 bp during the weak 2019–2021 regime.
  Long-SGD and short-SGD averages were both positive, at +6.50 and +8.74 bp.
- **04:00–08:00 has the highest average:** +9.07 bp in 2017–2023 and +18.56 bp in
  2024–2025. It was positive in eight of nine entry years, with 2020 the exception.
  The later period contains only 16 trades. The pooled mean's pointwise 95%
  three-month-block bootstrap interval is +5.81 to +18.05 bp. This broad window
  is heterogeneous: 05:00–06:00 alone had seven trades averaging −9.60 bp.
- **08:00–16:00 weakened recently:** both four-hour bins made money in the earlier
  sample but lost money in 2024–2025. Conversely, 16:00–20:00 improved. A blanket
  assumption that all daytime entries are persistently bad would miss this.
- The stronger windows are also stronger before costs, so the difference is not
  explained solely by cheaper spreads. Rankings remain similar after dividing
  each trade's net PnL by its absolute aggregate SGD-cross units.

One-common-quote execution delay reduces the 04:00–08:00 and 20:00–24:00 pooled
means to +10.70 and +6.26 bp. Their 2024–2025 means become +16.32 and +3.83 bp.
These are sensitivities for the same entry signals; delay is normally one minute
but can be longer across missing quotes.

## Day of the week

| Entry FX trading weekday | Trades | Mean gross, bp | Mean costs, bp | Mean net, bp | Mean net in 2024–2025, bp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monday | 134 | 10.11 | 4.63 | 5.48 | 8.24 |
| Tuesday | 158 | 8.02 | 3.46 | 4.56 | −2.39 |
| Wednesday | 154 | 7.73 | 4.82 | 2.91 | −4.00 |
| Thursday | 181 | 7.26 | 3.63 | 3.63 | −0.74 |
| Friday | 177 | 11.35 | 4.03 | **7.33** | **12.11** |

Friday looks strongest and Wednesday weakest on pooled means, but the weekday
effect is not statistically convincing after accounting for time of day and the
other controls below. Friday was positive in eight of nine entry years, but
Tuesday was stronger than Friday in the 2017–2023 pooled sample.

The calendar convention matters. Using literal Singapore weekdays, Monday–Friday
means lie in a narrow +3.89 to +5.24 bp range. The 20 trades entered on Singapore
Saturday, still Friday in New York, average +18.13 bp. Thus some of the apparent
FX-Friday advantage is associated with the late US-session hours rather than a
general Friday effect.

## Day within the month

| Entry FX calendar date | Trades | Mean gross, bp | Mean costs, bp | Mean net, bp | Mean net in 2024–2025, bp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–5 | 146 | 8.19 | 4.06 | 4.13 | 1.28 |
| 6–10 | 113 | 3.56 | 3.23 | **0.33** | 5.58 |
| 11–15 | 138 | 11.87 | 3.77 | **8.10** | **11.75** |
| 16–20 | 127 | 11.38 | 4.28 | 7.11 | 0.33 |
| 21–25 | 126 | 11.43 | 4.76 | 6.66 | 1.99 |
| 26–month-end | 154 | 6.59 | 4.26 | 2.33 | −1.98 |

The pooled data favor the middle of the month, especially dates 11–15. However,
some rankings reverse: dates 6–10 averaged −1.66 bp in 2017–2023 and +5.58 bp in
2024–2025. The broader date effect is not statistically convincing.

A separate turn-of-month check uses the first and last three Monday–Friday
business dates, without a holiday adjustment. Mean net PnL is +3.45 bp for the
first three, +5.77 bp for the middle and +1.91 bp for the last three. These
differences also have wide uncertainty. Individual dates 1–31 have only 16–34
trades each, so isolated best/worst dates are particularly fragile. All are saved
in the detailed output rather than selected as proposed entry filters.

## How much evidence is there?

The primary categories were specified before inspecting the grouped results:
six four-hour Singapore windows, five FX weekdays and six roughly five-day
calendar-date bins. A regression explains trade net PnL using all three calendar
dimensions simultaneously, with controls for entry year, long/short SGD direction
and aggregate units. Standard errors cluster by entry calendar month (103 months
with entries). The table below tests each whole categorical effect jointly.

| Dimension | Raw p-value, same quote | Holm-adjusted p-value | Holm-adjusted p-value, next quote |
| --- | ---: | ---: | ---: |
| Time of day | 0.016 | **0.049** | 0.143 |
| Weekday | 0.776 | 0.776 | 0.771 |
| Date within month | 0.315 | 0.630 | 0.746 |

The adjustment covers these three primary tests only; it does not account for
all earlier strategy research or the exploratory hourly, timezone and
turn-of-month breakdowns. The time effect is borderline under this specification
and weaker with delayed execution. It is a candidate for further testing, not
evidence of a validated trading restriction. These are all previously examined
historical dates, including the 2024–2025 comparison period.

Chart intervals resample three consecutive calendar-month blocks, 4,000 times,
preserving whole-trade outcomes within each month. They are pointwise intervals
for bucket means. A positive mean versus zero is different from proof that the
bucket outperforms the rest of the strategy; the corresponding difference
intervals are supplied separately.

## When PnL is earned versus when the trade enters

Daily marked-to-market attribution answers a different question. Across all
observed FX trading dates, including flat dates, **Tuesday earns the most net
PnL per day (+3.34 bp)**, followed by Friday (+1.80), Monday (+1.56), Thursday
(+0.79) and Wednesday (+0.75). A trade entered on Monday can earn most of its
profit on Tuesday. This is why entry-weekday rankings should not be read as
instructions to close existing positions on certain weekdays.

For the daily accrual view, dates 16–20 and 21–25 average +2.44 and +2.36 bp net
per day, compared with +0.64 bp on dates 6–10. These are descriptive marked-to-
market totals, not results from trading only those days. Intraday accrual has
not been separately attributed in this study; the hourly results concern entries.

## Reproduction and files

```sh
PYTHONDONTWRITEBYTECODE=1 poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_calendar_diagnostics.py
```

The runner uses the prepared minute cache and the saved original-spread trade
ledgers from the spread-sensitivity study. It checks flat-to-position-to-flat
episodes, unchanged component positions during each episode, valid quote
timestamps and gross/cost/net reconciliation to the original backtests under
both delays. Input hashes and maximum reconciliation errors are saved in
[verification.json](sgd_neer_results/calendar_diagnostics/verification.json).

- [Detailed grouped results](sgd_neer_results/calendar_diagnostics/entry_summary.csv)
- [Annual breakdowns](sgd_neer_results/calendar_diagnostics/entry_summary_annual.csv)
- [Bootstrap intervals](sgd_neer_results/calendar_diagnostics/bootstrap_intervals.csv)
- [Adjusted statistical tests](sgd_neer_results/calendar_diagnostics/adjusted_tests.csv)
- [Daily accrual attribution](sgd_neer_results/calendar_diagnostics/daily_accrual_summary.csv)
- [Chart PDF](sgd_neer_results/calendar_diagnostics/entry_calendar.pdf)
- [Full results and reconstructed episodes](sgd_neer_results/calendar_diagnostics/)
