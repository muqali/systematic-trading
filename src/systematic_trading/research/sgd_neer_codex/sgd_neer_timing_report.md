# SGD NEER: daily decision clocks versus rolling long-horizon signals

The prior multi-day tests made entry and price-based exit decisions at **16:00 New York**. Earlier intraday tests updated more frequently, but generally flattened before rollover. This follow-up tests other daily clocks and genuine every-minute signals with overnight and weekend holding.

All reported Sharpes deduct recorded bid/ask spreads and use daily PnL with √252 annualisation. **Forward-point adjustments remain excluded**, as requested. Primary execution uses the current common minute quote; one subsequent common quote is a sensitivity check.

## Experiment

The declared grid contains **80 configurations**:

- **40 daily-clock rules:** 00:00 / 04:00 / 08:00 / 12:00 / 16:00 New York × return reversal / EWMA equilibrium × 1 / 3 business days × 1.5 / 2 entry z. Only the clock changes relative to the prior daily methodology. All eight 16:00 controls reproduce the previous daily PnL exactly.
- **20 continuously reviewed rules:** return reversal / EWMA equilibrium × 6 / 24 / 72 / 120 / 240 market hours × 1.5 / 2 entry z. A new price decision can occur at **any available minute**, with no fixed entry hour.
- **20 matched rolling controls:** exactly the same rolling features and sizing, but entry and price-based exits can occur only at 16:00. This isolates decision frequency from changes in normalization and horizon definitions.

A market hour means 60 scheduled minutes in the source loader's Sunday 18:00–Friday 17:00 New York trading week. **24 market hours skip the weekend closure**; they are not 24 elapsed calendar hours. Missing market minutes remain on the grid, so outages cannot shorten a horizon. Quotes are not interpolated. Existing positions earn every intervening gap return.

The rolling return signal is `log_index(t) - log_index(t-H)`, less its prior EWM mean, divided by its prior empirical EWM standard deviation (63-market-day half life, minimum 40 market days of available observations). The equilibrium signal uses deviation from the prior H-hour-half-life EWMA level, standardized by prior empirical deviation dispersion. These are proxy-basket signals, not observations of MAS's unpublished band midpoint.

Enter when |z|≥1.5 or 2 and half the absolute displacement exceeds 1.5 full basket spreads. The half-reversion edge is a heuristic, not a calibrated forecast. Exit near/crossing the mean (0.25z) or after 2H market hours. Size once at entry: 5% annual volatility target divided by prior empirical 24-market-hour return volatility (20-market-day half life), capped at three units. There is no pyramiding or continuous resizing. Rolling return observations overlap; they are not counted as independent days of statistical evidence.

Both continuous and clocked rolling rules enforce the known timeout every minute, including missing-quote minutes. A requested exit waits for the next available quote. New entry is blocked until that exit has had an opportunity to fill. Daily-clock controls retain the prior business-day timeout. No rule automatically flattens overnight or over weekends.

## Does the daily time matter?

Yes. One-day rules show substantial clock sensitivity. The following table is **2024–2025 net Sharpe**, with identical daily-rule methods at each clock:

| hour_ny | return, z=1.5 | return, z=2 | equilibrium, z=1.5 | equilibrium, z=2 |
| --- | --- | --- | --- | --- |
| 0.000 | 0.351 | 0.547 | 0.452 | -0.136 |
| 4.000 | 0.551 | 1.132 | 0.475 | 0.341 |
| 8.000 | 0.164 | 0.553 | 0.161 | 0.569 |
| 12.000 | 0.023 | -0.151 | 0.426 | 0.319 |
| 16.000 | 0.729 | 0.544 | 1.286 | 0.674 |

For example, one-day return reversal at 2z reaches **1.132** at 04:00 versus **0.544** at 16:00. Its development Sharpe at 04:00 is **0.244**, so the stronger final-period score alone does not establish the best clock. Full three-day clock results are included in the notebook, figure and CSV.

## Does removing the fixed decision time help?

Below, `continuous` reviews every minute and `clock16` reviews the **same rolling signal** only at 16:00. Values are **2024–2025 net Sharpe**.

| kind | horizon_hours | entry_z | clock16 | continuous |
| --- | --- | --- | --- | --- |
| equilibrium | 6 | 1.500 | 0.524 | 0.255 |
| equilibrium | 6 | 2.000 | 0.805 | 0.309 |
| equilibrium | 24 | 1.500 | 0.860 | 0.907 |
| equilibrium | 24 | 2.000 | 0.857 | 0.246 |
| equilibrium | 72 | 1.500 | 0.028 | 0.785 |
| equilibrium | 72 | 2.000 | -0.197 | 0.172 |
| equilibrium | 120 | 1.500 | -0.066 | 0.518 |
| equilibrium | 120 | 2.000 | 0.053 | 0.256 |
| equilibrium | 240 | 1.500 | -0.267 | 0.381 |
| equilibrium | 240 | 2.000 | -0.439 | -0.035 |
| return | 6 | 1.500 | 0.030 | -1.906 |
| return | 6 | 2.000 | 0.228 | -0.334 |
| return | 24 | 1.500 | 1.057 | -0.698 |
| return | 24 | 2.000 | 0.403 | -0.228 |
| return | 72 | 1.500 | 0.048 | 0.303 |
| return | 72 | 2.000 | -0.579 | 0.045 |
| return | 120 | 1.500 | -0.621 | -0.147 |
| return | 120 | 2.000 | -0.515 | -0.627 |
| return | 240 | 1.500 | 0.600 | 0.578 |
| return | 240 | 2.000 | 0.645 | 0.742 |

The highest continuous final-period score is **0.907**, from `rolling_equilibrium_24h_z1.5_continuous`, with **189** completed trades. This is the maximum of a disclosed exploratory grid, not a model chosen using unseen evaluation data.

The 24-hour equilibrium rule at 1.5z also ranks first using only the declared development/validation criterion: Sharpe **1.454** in 2017–2022, **1.259** in 2021–2022 and **2.045** in 2023. Its 2024 and 2025 Sharpes are **1.146** and **0.677**. This is a more consistent candidate than choosing an entry clock from the final two years alone. However, its 2024–2025 Sharpe falls from **0.907 to 0.321** with one subsequent quote of execution delay; adding **0.25 bp per leg** also lowers it to **0.321**, and doubling spreads makes it **−1.162**. The 95% block-bootstrap interval is **[−0.393, 2.224]**. Its economic edge remains execution-sensitive.

For the 24-hour return rule at 2z, continuous monitoring completes **117** trades versus **27** for the 16:00 control. Gross Sharpe is **1.478** versus **0.718**, but net Sharpe is **-0.228** versus **0.403**. This directly measures the opportunity/turnover tradeoff from more frequent review.

## Stability and selection

| policy | development | recent_development | validation | 2024 | 2025 | evaluation | all_2017_2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clock16_return_1d_z1.5 | 0.901 | 0.929 | 0.392 | 1.263 | 0.252 | 0.729 | 0.811 |
| clock16_return_1d_z2 | 0.930 | 1.390 | 0.316 | 0.966 | 0.215 | 0.544 | 0.778 |
| clock04_return_1d_z2 | 0.244 | -0.227 | 0.264 | 1.500 | 0.864 | 1.132 | 0.401 |
| clock16_equilibrium_1d_z1.5 | 0.058 | 0.244 | 0.794 | 1.940 | 0.658 | 1.286 | 0.380 |
| rolling_return_24h_z2_continuous | 0.901 | 0.958 | 1.475 | 0.470 | -1.031 | -0.228 | 0.713 |
| rolling_return_24h_z2_clock16 | 0.927 | 1.007 | 0.336 | 0.964 | -0.097 | 0.403 | 0.738 |
| rolling_return_72h_z2_continuous | 0.492 | 1.405 | 0.774 | 0.480 | -0.377 | 0.045 | 0.406 |
| rolling_return_120h_z2_continuous | 0.238 | 0.358 | 1.160 | -0.241 | -0.934 | -0.627 | 0.095 |
| rolling_return_240h_z2_continuous | 0.259 | 1.383 | 2.250 | 1.108 | 0.401 | 0.742 | 0.510 |
| rolling_equilibrium_24h_z2_continuous | 0.924 | 1.203 | 0.988 | 0.359 | 0.123 | 0.246 | 0.785 |
| rolling_equilibrium_24h_z1.5_continuous | 1.454 | 1.259 | 2.045 | 1.146 | 0.677 | 0.907 | 1.397 |
| rolling_equilibrium_6h_z2_continuous | 1.064 | 1.107 | 1.876 | 0.393 | 0.218 | 0.309 | 0.963 |

The predeclared shortlist criterion uses the minimum Sharpe over 2017–2022, 2021–2022 and 2023, requiring it to be positive, with at least ten completed 2023 trades and forty active 2023 days. **Shortlist: `rolling_equilibrium_24h_z1.5_continuous`, `rolling_equilibrium_6h_z2_continuous`, `rolling_equilibrium_24h_z2_continuous`.** No 2024–2025 score enters this ranking. The complete leaderboard is saved, including rejected configurations. No new ensemble was fitted.

The 2024–2025 data were already examined in earlier experiments; these results are **exploratory, not a fresh holdout**. Trying multiple clocks and horizons increases selection risk. In particular, a stronger recent clock can coexist with poor earlier performance. The block-bootstrap intervals below use 20-day blocks and 2,000 draws; they do not correct for the strategy search or prior experiments.

| policy | low | median | high | zero_vol_draws |
| --- | --- | --- | --- | --- |
| clock16_return_1d_z1.5 | -0.366 | 0.723 | 1.829 | 0 |
| clock16_return_1d_z2 | -0.832 | 0.591 | 1.771 | 0 |
| clock04_return_1d_z2 | -0.411 | 1.180 | 2.622 | 0 |
| clock16_equilibrium_1d_z1.5 | 0.091 | 1.302 | 2.388 | 0 |
| rolling_return_24h_z2_continuous | -1.446 | -0.238 | 1.144 | 0 |
| rolling_return_24h_z2_clock16 | -0.756 | 0.428 | 1.427 | 0 |
| rolling_return_72h_z2_continuous | -1.435 | 0.021 | 1.491 | 0 |
| rolling_return_120h_z2_continuous | -1.713 | -0.624 | 0.550 | 0 |
| rolling_return_240h_z2_continuous | -0.367 | 0.748 | 2.018 | 0 |
| rolling_equilibrium_24h_z2_continuous | -1.067 | 0.290 | 1.642 | 0 |
| rolling_equilibrium_24h_z1.5_continuous | -0.393 | 0.940 | 2.224 | 0 |
| rolling_equilibrium_6h_z2_continuous | -0.677 | 0.335 | 1.384 | 0 |

## Execution sensitivity

| policy | 0 | 1 |
| --- | --- | --- |
| clock04_return_1d_z2 | 1.132 | 1.207 |
| clock16_equilibrium_1d_z1.5 | 1.286 | 1.289 |
| clock16_return_1d_z1.5 | 0.729 | 0.687 |
| clock16_return_1d_z2 | 0.544 | 0.541 |
| rolling_equilibrium_24h_z1.5_continuous | 0.907 | 0.321 |
| rolling_equilibrium_24h_z2_continuous | 0.246 | -0.019 |
| rolling_equilibrium_6h_z2_continuous | 0.309 | -0.652 |
| rolling_return_120h_z2_continuous | -0.627 | -0.545 |
| rolling_return_240h_z2_continuous | 0.742 | 0.760 |
| rolling_return_24h_z2_clock16 | 0.403 | 0.408 |
| rolling_return_24h_z2_continuous | -0.228 | -0.347 |
| rolling_return_72h_z2_continuous | 0.045 | 0.038 |

| policy | base | extra_0.25bp_per_leg | spread_x2 |
| --- | --- | --- | --- |
| clock04_return_1d_z2 | 1.132 | 0.976 | 0.858 |
| clock16_equilibrium_1d_z1.5 | 1.286 | 1.117 | 0.882 |
| clock16_return_1d_z1.5 | 0.729 | 0.517 | 0.190 |
| clock16_return_1d_z2 | 0.544 | 0.433 | 0.317 |
| rolling_equilibrium_24h_z1.5_continuous | 0.907 | 0.321 | -1.162 |
| rolling_equilibrium_24h_z2_continuous | 0.246 | -0.087 | -0.957 |
| rolling_equilibrium_6h_z2_continuous | 0.309 | -0.765 | -2.669 |
| rolling_return_120h_z2_continuous | -0.627 | -0.714 | -0.904 |
| rolling_return_240h_z2_continuous | 0.742 | 0.687 | 0.586 |
| rolling_return_24h_z2_clock16 | 0.403 | 0.274 | 0.077 |
| rolling_return_24h_z2_continuous | -0.228 | -0.717 | -1.834 |
| rolling_return_72h_z2_continuous | 0.045 | -0.132 | -0.503 |

Delay counts common valid quote rows, so it can exceed one elapsed minute across an outage. Additional slippage is charged per unit of absolute leg turnover. The historical minute caches may contain quotes forward filled by up to ten minutes by the exporter; this experiment adds no filling. No financing, commission or impact is included in the base case. PnL follows the existing fixed log-return notional convention rather than a cash-settlement ledger.

## Data coverage

| index | count | sum | mean | mode | hour_ny |
| --- | --- | --- | --- | --- | --- |
| 2023 | 260 | 258 | 0.992 | daily_clock | 0.000 |
| 2023 | 260 | 258 | 0.992 | daily_clock | 4.000 |
| 2023 | 260 | 146 | 0.562 | daily_clock | 8.000 |
| 2023 | 260 | 147 | 0.565 | daily_clock | 12.000 |
| 2023 | 260 | 148 | 0.569 | daily_clock | 16.000 |
| 2023 | 371640 | 287566 | 0.774 | rolling_grid | nan |

Clock comparisons can also be affected by which exact marks are missing. In 2023, common minute coverage is materially incomplete, and missing daily marks suppress entries. The rolling strategies can react at other available minutes, but their historical endpoints can still be unavailable. Per-year feature usability is disclosed in `coverage.csv`. No missing date or minute is silently compressed.

## Reproduce

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_timing.py
poetry run pytest -q tests/test_sgd_neer_timing.py tests/test_sgd_neer_multiday.py tests/test_sgd_neer_backtest.py tests/test_resampled_data_loader.py
```

The runner uses `tmp/sgd_neer_m1_research/data.pkl`, which was prepared from five-pair 2016–2025 quotes with `freq = pd.Timedelta(minutes=1)`. If absent, use `run_sgd_neer_minute.py --stage prepare` first. Tests cover causal feature/signal prefixes, missing endpoints, weekend grids, off-clock entries, timeout fills and accounting equivalence. All 80 rules, daily PnL, actual fill ledgers, entry-hour counts, spreads/slippage stress, protocol and source/input manifests are in `sgd_neer_results/timing`.

[Executed notebook](sgd_neer_timing.ipynb) · [Previous multi-day study](sgd_neer_multiday_report.md)
