# SGD NEER: attribute index extremes and trade selected constituents

The whole-index assumption can be relaxed. This experiment retains the existing index mean-reversion trigger and exit, attributes the deviation to SGD/currency crosses, and trades a subset chosen at entry. Selections and notionals remain fixed until the original index exit. Every selected synthetic cross retains its correct USDSGD hedge.

The most useful result is modest: for the **24-market-hour EWMA / 1.5z** index trigger, retaining all aligned contributors and sizing to the original estimated entry risk reduced 2024–2025 spread costs by **33.5%**. Net Sharpe changed from **0.907 to 0.991**. With a one-quote execution delay the comparison is **0.321 versus 0.530**. Concentrating into the single largest contributor generally discarded useful reversion exposure.

All results deduct recorded bid/ask spreads. **Forward-point adjustments remain excluded** as requested. The 2024–2025 period has already been examined in preceding research; this is an exploratory extension, not a fresh holdout.

## How the index deviation is attributed

Construct log levels for SGD/USD, SGD/EUR, SGD/AUD, SGD/GBP and SGD/JPY from the five quoted markets. Let `x_i` be a cross's log level and `w_i` its normalized index weight. For the equilibrium signal:

`d_i(t) = x_i(t) - prior_EWMA_H(x_i)`

`contribution_i(t) = w_i * d_i(t)`

The same linear operation is applied to every cross on the same market-minute grid, so **index deviation = sum of component contributions**. This equality was checked over the full history. For return reversal, use the H-hour return minus its prior 63-day-half-life average instead; the same additivity holds.

A component is **aligned** when its contribution has the same sign as the index deviation. If the index is high, trade short SGD against the selected currencies; if low, trade long SGD. This is a historical price attribution, not proof of an economic cause or a forecast that the driver must reverse. An index can recover through changes in other constituents while its original driver continues trending.

Illustrative entry: **2024-01-03 08:07:00-05:00**, index deviation **14.27 bp**. This is the first 2024–2025 index entry with at least two aligned crosses and one opposing cross; it is selected without consulting future outcomes.

| currency | weighted_deviation_bp | aligned |
| --- | --- | --- |
| USD | -10.875 | False |
| EUR | 9.212 | True |
| AUD | 7.405 | True |
| GBP | 0.198 | True |
| JPY | 8.325 | True |

Synthetic-cross attribution shares the USDSGD movement across all constituents. A complementary decomposition into the **actual quoted legs** identifies the largest absolute deviation contribution as USDSGD at **74.1%** of this trigger's 2024–2025 entries. `quoted_leg_attribution.csv` contains that decomposition. Selecting a large SGD/EUR contribution does not, by itself, establish that EURUSD caused the move.

## Declared subset rules and sizing

Four existing index triggers are tested: equilibrium 24h at 1.5z and 2z, equilibrium 72h at 1.5z, and return reversal 24h at 2z. They use the prior every-minute policy, the full-basket entry spread hurdle, overnight/weekend holding, near-mean exit and maximum hold of 2H market hours. All original signal events reproduce exactly. Subsets cannot create new entries at times when the full basket did not enter.

- **Full:** the original five SGD crosses.
- **Aligned:** every contributor with the same sign as the index deviation.
- **Top 1 / top 2:** up to one or two largest aligned weighted contributions.
- **Coverage 80%:** the smallest ranked aligned set accounting for 80% of the positive contribution total. The denominator is the positive total, not the net index deviation after offsets.
- **Cost-aware top 2:** at most two aligned crosses whose assumed half-reversion exceeds 1.5 synthetic full spreads, ranked by weighted contribution per weighted spread. The half-reversion assumption remains a heuristic, not an estimated conditional return.

For each subset, three sizing conventions distinguish true cost efficiency from changes in exposure:

1. **Retained weights:** keep the original constituent weights and remove excluded components. Aggregate SGD exposure falls. Risk need not fall proportionally because excluded constituents can have provided diversification.
2. **Normalized weights:** renormalize the selected weights to sum to one, retaining original aggregate SGD exposure. Concentrated currency risk can rise substantially.
3. **Risk matched:** normalize the weights, then scale size to match the original basket's estimated entry volatility. Covariance uses only prior 24-market-hour returns, with a 20-day EWM half-life and 40-day minimum history. Total cross units are capped at three. This matches estimated entry risk, not future realized volatility.

There are **64 configurations**: four index triggers × (one full-basket control + five selectors × three sizing conventions). No constituents are switched while a trade is open. The common five-pair quote grid is retained for every portfolio so coverage does not change between strategies.

## Primary comparison: 24h equilibrium at 1.5z

**2024–2025, immediate execution.** Cost savings are relative to the same full-index trigger; volatility is realized annualized net volatility.

| rule | sizing | sharpe | mean_selected_count | mean_exante_risk_ratio | cost_saved_pct | annual_vol_pct |
| --- | --- | --- | --- | --- | --- | --- |
| full | retained | 0.907 | 5.000 | 1.000 | 0.000 | 3.220 |
| aligned | retained | 0.930 | 3.423 | 1.104 | 29.208 | 3.231 |
| aligned | normalized | 0.948 | 3.423 | 1.700 | 0.923 | 5.023 |
| aligned | risk_matched | 0.991 | 3.423 | 1.000 | 33.547 | 2.934 |
| top1 | risk_matched | 0.076 | 1.000 | 1.000 | 68.794 | 2.656 |
| top2 | risk_matched | 0.737 | 1.979 | 1.000 | 52.333 | 2.749 |
| coverage80 | risk_matched | 0.850 | 2.032 | 1.000 | 51.115 | 2.716 |
| cost_top2 | risk_matched | 0.705 | 1.942 | 1.000 | 55.731 | 2.729 |

The aligned subset contains **3.42 crosses on average**. Its estimated entry risk matches the baseline by construction, while its realized net volatility is **2.93%**, versus **3.22%** for the original basket. Reduced diversification means that even retained original weights can carry more estimated risk than the full basket. The risk-matched comparison is therefore more informative than a cost-only ranking.

## Why fewer components do not automatically mean lower costs

For selected cross exposures `a_i`, the combined **USDSGD loading is −sum(a_i)**. For example, SGD/EUR and SGD/GBP each require their own contribution to that common hedge; the contributions must be accumulated before costing the actual market trades.

USDSGD accounts for **80.3%** of the full strategy's 2024–2025 spread bill. Keeping aggregate SGD exposure unchanged therefore leaves the dominant cost intact. For the top-one normalized portfolio, spread costs are **98.5%** of baseline while mean estimated entry risk rises to **3.47×** baseline. The savings from risk-matched aligned selection mainly arise because a smaller total SGD notional reaches the same estimated portfolio risk after changing the currency mix.

## Do the contributors actually reverse?

At the original index exit, the descriptive outcomes for the 1.5z trigger in 2024–2025 are:

| aligned | cross_observations | gross_bp | net_bp | gross_win_fraction |
| --- | --- | --- | --- | --- |
| False | 298 | 2.268 | -0.922 | 0.604 |
| True | 647 | 6.280 | 3.073 | 0.655 |

These are equally weighted, per-unit synthetic-cross returns, including each cross's actual two-leg round-trip spread. They are not portfolio Sharpes. The observations share **189 index trade episodes**, so they are not independent samples. Outcomes are calculated only for diagnosis and are never passed into selection.

Opposing components still had positive gross mean returns in the index trade direction on this sample, but their average return did not cover their spread cost. The strongest weighted contributor was not uniformly the most profitable one:

| contribution_rank | cross_observations | gross_bp | net_bp |
| --- | --- | --- | --- |
| 1 | 189 | 4.721 | 1.860 |
| 2 | 189 | 7.743 | 4.718 |
| 3 | 189 | 5.515 | 2.018 |
| 4 | 189 | 5.423 | 2.125 |
| 5 | 189 | 1.673 | -1.654 |

## Stability, selection and uncertainty

| policy | development | recent_development | validation | 2024 | 2025 | evaluation | all_2017_2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equilibrium_24h_z1.5__full__retained | 1.454 | 1.259 | 2.045 | 1.146 | 0.677 | 0.907 | 1.397 |
| equilibrium_24h_z1.5__aligned__retained | 1.392 | 1.396 | 2.388 | 0.585 | 1.352 | 0.930 | 1.378 |
| equilibrium_24h_z1.5__aligned__normalized | 1.442 | 1.743 | 2.375 | 0.675 | 1.307 | 0.948 | 1.419 |
| equilibrium_24h_z1.5__aligned__risk_matched | 1.368 | 1.359 | 2.324 | 0.464 | 1.514 | 0.991 | 1.369 |
| equilibrium_24h_z1.5__top1__risk_matched | 0.814 | 1.455 | 0.449 | -0.651 | 0.803 | 0.076 | 0.631 |
| equilibrium_24h_z1.5__top2__risk_matched | 1.178 | 1.583 | 1.495 | 0.427 | 1.021 | 0.737 | 1.111 |
| equilibrium_24h_z1.5__coverage80__risk_matched | 1.263 | 1.369 | 2.040 | -0.171 | 1.956 | 0.850 | 1.248 |
| equilibrium_24h_z1.5__cost_top2__risk_matched | 1.048 | 1.338 | 1.323 | -0.342 | 1.787 | 0.705 | 1.003 |

The risk-matched aligned rule's 2024–2025 improvement is **0.084 Sharpe**, with a paired 20-day-block bootstrap interval of **[-0.750, 0.809]** using 2,000 shared resamples. This does not establish a reliable Sharpe improvement. Its development Sharpe is **1.368**, versus **1.454** for the parent, so it fails the strict requirement to improve all development/validation periods. Its appeal is lower cost exposure with broadly similar historical performance, not a newly validated optimum.

The development-only shortlist requires positive Sharpe and improvement over the corresponding full-index parent in **all three** selection periods (2017–2022, 2021–2022, 2023), at least ten validation trades and forty active validation days. Rank the minimum Sharpe improvement. The selected configurations and subsequent results are:

| policy | sharpe | cost_ratio | mean_exante_risk_ratio | development | recent_development | validation |
| --- | --- | --- | --- | --- | --- | --- |
| equilibrium_24h_z2__coverage80__normalized | 0.581 | 0.975 | 2.162 | 1.255 | 1.709 | 2.179 |
| equilibrium_24h_z2__top2__normalized | 0.414 | 0.976 | 2.103 | 1.213 | 2.000 | 1.334 |
| equilibrium_24h_z2__aligned__normalized | 0.424 | 0.996 | 1.556 | 1.162 | 2.178 | 2.214 |

These selected rules use the 2z trigger and normalized sizing. They show little reduction in absolute costs and increase estimated risk, so the development ranking alone does not fulfill the cost-saving objective. Full results for all four triggers and all sizing rules follow below. No ensemble or further parameters were fitted after seeing evaluation results.

## Delay and spread sensitivity

| policy | 0 | 1 |
| --- | --- | --- |
| equilibrium_24h_z1.5__aligned__normalized | 0.948 | 0.539 |
| equilibrium_24h_z1.5__aligned__retained | 0.930 | 0.526 |
| equilibrium_24h_z1.5__aligned__risk_matched | 0.991 | 0.530 |
| equilibrium_24h_z1.5__cost_top2__risk_matched | 0.705 | 0.379 |
| equilibrium_24h_z1.5__coverage80__risk_matched | 0.850 | 0.456 |
| equilibrium_24h_z1.5__full__retained | 0.907 | 0.321 |
| equilibrium_24h_z1.5__top1__risk_matched | 0.076 | -0.200 |
| equilibrium_24h_z1.5__top2__risk_matched | 0.737 | 0.365 |

| policy | base | extra_0.25bp_per_leg | spread_x2 |
| --- | --- | --- | --- |
| equilibrium_24h_z1.5__aligned__normalized | 0.948 | 0.570 | -0.371 |
| equilibrium_24h_z1.5__aligned__retained | 0.930 | 0.514 | -0.528 |
| equilibrium_24h_z1.5__aligned__risk_matched | 0.991 | 0.570 | -0.519 |
| equilibrium_24h_z1.5__cost_top2__risk_matched | 0.705 | 0.406 | -0.380 |
| equilibrium_24h_z1.5__coverage80__risk_matched | 0.850 | 0.515 | -0.358 |
| equilibrium_24h_z1.5__full__retained | 0.907 | 0.321 | -1.162 |
| equilibrium_24h_z1.5__top1__risk_matched | 0.076 | -0.147 | -0.711 |
| equilibrium_24h_z1.5__top2__risk_matched | 0.737 | 0.411 | -0.425 |

Delay is one subsequent common valid quote, usually one minute but potentially longer across missing data. The aligned subset retains more Sharpe under the displayed execution stresses, but doubled spreads still erase its profit. Neither subset selection nor covariance sizing removes the quote-age and execution assumptions.

## Complete grid: 2024–2025 net Sharpe

| trigger | rule | normalized | retained | risk_matched |
| --- | --- | --- | --- | --- |
| equilibrium_24h_z1.5 | aligned | 0.948 | 0.930 | 0.991 |
| equilibrium_24h_z1.5 | cost_top2 | 0.539 | 0.831 | 0.705 |
| equilibrium_24h_z1.5 | coverage80 | 0.690 | 0.842 | 0.850 |
| equilibrium_24h_z1.5 | full | nan | 0.907 | nan |
| equilibrium_24h_z1.5 | top1 | 0.360 | 0.351 | 0.076 |
| equilibrium_24h_z1.5 | top2 | 0.637 | 0.744 | 0.737 |
| equilibrium_24h_z2 | aligned | 0.424 | 0.238 | 0.267 |
| equilibrium_24h_z2 | cost_top2 | 0.547 | 0.197 | 0.121 |
| equilibrium_24h_z2 | coverage80 | 0.581 | -0.036 | 0.014 |
| equilibrium_24h_z2 | full | nan | 0.246 | nan |
| equilibrium_24h_z2 | top1 | 0.418 | 0.161 | -0.094 |
| equilibrium_24h_z2 | top2 | 0.414 | 0.205 | 0.228 |
| equilibrium_72h_z1.5 | aligned | 0.555 | 0.266 | 0.161 |
| equilibrium_72h_z1.5 | cost_top2 | 0.368 | 0.101 | -0.042 |
| equilibrium_72h_z1.5 | coverage80 | 0.800 | 0.622 | 0.451 |
| equilibrium_72h_z1.5 | full | nan | 0.785 | nan |
| equilibrium_72h_z1.5 | top1 | 0.834 | 0.787 | 0.634 |
| equilibrium_72h_z1.5 | top2 | 0.548 | 0.319 | 0.125 |
| return_24h_z2 | aligned | 0.540 | 0.050 | -0.119 |
| return_24h_z2 | cost_top2 | -0.331 | -0.341 | -0.538 |
| return_24h_z2 | coverage80 | 0.257 | -0.189 | -0.475 |
| return_24h_z2 | full | nan | -0.228 | nan |
| return_24h_z2 | top1 | 0.108 | -0.190 | -0.444 |
| return_24h_z2 | top2 | -0.180 | -0.205 | -0.346 |

## Accounting, coverage and reproduction

Trades are costed on the net USD-quoted legs after synthetic-cross translation. The sparse engine splits the minute history at fills and New York trading-date boundaries and telescopes price changes under constant positions. It retains every held-position gap return and charges final liquidation. Tests compare it against the general five-leg engine at delays 0/1/15, including missing-quote exits and weekend gaps. All four full controls reproduce the prior research at both tested delays.

The input is the existing five-pair 2016–2025 cache prepared with `freq = pd.Timedelta(minutes=1)`. Missing market minutes remain in the feature grid; weekend closure is excluded. 2023 coverage is incomplete. Historical cache quotes may already be forward filled by the exporter for up to ten minutes; this experiment adds no filling. Financing, commission and impact are excluded; extra slippage is shown separately. PnL uses the existing fixed log-return notional convention rather than a cash-settlement ledger. Bootstrap intervals do not adjust for the growing strategy search.

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_subset.py
poetry run pytest -q tests/test_sgd_neer_subset.py tests/test_sgd_neer_timing.py tests/test_sgd_neer_multiday.py tests/test_sgd_neer_backtest.py tests/test_resampled_data_loader.py
```

The 47 related tests passed. Definitions are in `sgd_neer_subset.py`: `component_deviations`, `select_crosses`, `subset_targets`, and `PairAccountant`. The runner fixes the four triggers and subset grid. Results include all daily PnL, pair fill ledgers, per-entry selections/deviations/risk estimates, component outcomes, paired bootstrap intervals and source/input hashes.

[Executed notebook](sgd_neer_subset.ipynb) · [Previous rolling signal study](sgd_neer_timing_report.md)
