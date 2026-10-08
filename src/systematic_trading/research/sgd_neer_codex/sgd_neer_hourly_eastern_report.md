# Hourly entry PnL in US/Eastern

The 24h / 1.5z equilibrium strategy with aligned constituents, risk-matched sizing and a 48h holding limit is unchanged. Results use original spreads, indicators initialized in 2017 and all 804 completed trades from the notebook-matched 2017–2025 sample. Financing remains excluded.

Timestamps are converted from UTC to `US/Eastern`, preserving EST/EDT transitions. Hour 08 means 08:00 inclusive to 09:00 exclusive. The full entry-to-exit PnL of all legs is assigned to the **entry signal hour**; this is not PnL earned within that hour. For the one-common-quote delay sensitivity, the grouping remains the original signal hour.

Figures are basis points of the existing position-weighted portfolio PnL convention. Pooled average net PnL is +4.80 bp/trade. Recent-period counts are included because some apparently large means depend on very few observations.

![Hourly entry results](sgd_neer_results/hourly_eastern/hourly_eastern.png)

| Eastern entry hour | Trades | Mean gross, bp | Mean costs, bp | Mean net, bp | Win rate | 2024–25 trades | 2024–25 mean net, bp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00:00–01:00 | 29 | 2.84 | 5.27 | -2.43 | 62.1% | 9 | -11.72 |
| 01:00–02:00 | 23 | 10.26 | 7.41 | +2.85 | 60.9% | 6 | -20.13 |
| 02:00–03:00 | 49 | 10.12 | 3.88 | +6.24 | 67.3% | 9 | +6.63 |
| 03:00–04:00 | 63 | 6.15 | 4.56 | +1.60 | 66.7% | 13 | -19.19 |
| 04:00–05:00 | 54 | 2.56 | 4.25 | -1.69 | 63.0% | 14 | +2.10 |
| 05:00–06:00 | 45 | 1.81 | 3.09 | -1.27 | 66.7% | 11 | +16.70 |
| 06:00–07:00 | 30 | 8.01 | 3.55 | +4.46 | 76.7% | 8 | +15.90 |
| 07:00–08:00 | 21 | 15.53 | 3.74 | +11.79 | 81.0% | 4 | +4.63 |
| 08:00–09:00 | 42 | 8.64 | 3.52 | +5.12 | 57.1% | 10 | +9.59 |
| 09:00–10:00 | 44 | 10.63 | 4.22 | +6.41 | 75.0% | 8 | +6.30 |
| 10:00–11:00 | 72 | 12.46 | 4.32 | +8.15 | 76.4% | 16 | +3.35 |
| 11:00–12:00 | 40 | 19.27 | 4.07 | +15.20 | 87.5% | 3 | +19.62 |
| 12:00–13:00 | 28 | 2.85 | 2.86 | -0.01 | 67.9% | 8 | +7.28 |
| 13:00–14:00 | 18 | 5.63 | 3.91 | +1.72 | 72.2% | 5 | +12.76 |
| 14:00–15:00 | 16 | 9.60 | 4.38 | +5.23 | 68.8% | 4 | +14.99 |
| 15:00–16:00 | 9 | 13.15 | 2.47 | +10.68 | 77.8% | 1 | +15.91 |
| 16:00–17:00 | 14 | 10.20 | 5.56 | +4.64 | 64.3% | 3 | +19.39 |
| 17:00–18:00 | 3 | 11.53 | 5.04 | +6.49 | 66.7% | 0 | — |
| 18:00–19:00 | 30 | 20.32 | 4.61 | +15.71 | 86.7% | 12 | +18.57 |
| 19:00–20:00 | 22 | 4.98 | 6.46 | -1.47 | 59.1% | 4 | -16.61 |
| 20:00–21:00 | 30 | 8.03 | 3.55 | +4.48 | 73.3% | 6 | +2.29 |
| 21:00–22:00 | 52 | 5.64 | 3.62 | +2.02 | 63.5% | 16 | -1.83 |
| 22:00–23:00 | 35 | 10.56 | 3.32 | +7.24 | 62.9% | 8 | -2.74 |
| 23:00–24:00 | 35 | 11.31 | 2.57 | +8.74 | 68.6% | 11 | +2.24 |

## Reading the pattern

- **18:00–19:00** and **11:00–12:00** have the highest pooled net means, at +15.71 and +15.20 bp, from 30 and 40 trades. Both earlier- and later-period means are positive. The 11:00 hour has only three trades in 2024–2025, so its recent result is particularly imprecise.
- **08:00–12:00** is a reasonably populated area of positive pooled means. Each of its four hours is also positive in 2024–2025. This describes a candidate pattern, not an evaluated entry restriction.
- **00:00, 04:00, 05:00 and 19:00** have negative pooled net means; **12:00** is approximately flat. The weak hours are not all stable: 04:00 and 05:00 turn positive in 2024–2025, whereas 01:00 and 03:00 become sharply negative.
- **15:00–18:00 is sparse:** the three hourly buckets contain only nine, 14 and three entries in the full sample. There are no 17:00 entries in 2024–2025. Strong-looking means there should be read with the counts.
- The previous fixed Singapore windows cannot be shifted to one constant Eastern interval: Singapore has no daylight-saving change, while Eastern time does. All hourly grouping here is recomputed directly from the original timestamps.

## Delayed execution and uncertainty

With one-common-quote delayed execution, pooled net means at 11:00 and 18:00 are +13.24 and +15.29 bp, respectively. A common-quote delay is usually one minute and can be longer across missing quotes. The detailed table includes every hour under both delays.

An exploratory regression jointly tests the 24 hourly categories, controlling for entry year, SGD direction, position units, FX weekday and the six calendar-date bins. Standard errors cluster by entry calendar month. These are global hourly-effect tests, not significance claims for a selected hour:

| Execution | Hourly-effect F statistic | Raw p-value |
| --- | ---: | ---: |
| Same quote | 1.863 | 0.0185 |
| Next common quote | 1.860 | 0.0187 |

The hourly split is a follow-up to an already inspected calendar analysis. These p-values do not adjust for that broader search. Bootstrap intervals are pointwise 95% intervals from 4,000 circular resamples of three-calendar-month blocks; they do not establish that the best observed hour will remain best. Full and recent samples overlap, and neither is a new untouched holdout.

## Reproduction and verification

The runner reuses the previously reconciled completed-trade ledgers. It checks that every hourly grouping contains all 804 episodes and sums to the original gross, cost and net totals, and verifies US/Eastern against America/New_York hour by hour. No positions or strategy parameters are changed.

```sh
PYTHONDONTWRITEBYTECODE=1 poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_hourly_eastern.py
```

- [Hourly results, including direction and execution-delay splits](sgd_neer_results/hourly_eastern/hourly_summary.csv)
- [Annual hourly results](sgd_neer_results/hourly_eastern/hourly_annual.csv)
- [Bootstrap intervals](sgd_neer_results/hourly_eastern/bootstrap_intervals.csv)
- [Statistical tests](sgd_neer_results/hourly_eastern/adjusted_tests.csv)
- [Verification and source hashes](sgd_neer_results/hourly_eastern/verification.json)
- [Chart PDF](sgd_neer_results/hourly_eastern/hourly_eastern.pdf)
