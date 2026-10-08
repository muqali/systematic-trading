# SGD NEER: holding-time limits and winning parameters

**Changing only the original 24h strategy's holding cap does not improve its
2024–2025 performance.** Net Sharpe falls from 0.991 at 48h to 0.728 with no
time exit. The weak 2019–2021 period improves slightly but remains negative.
The pre-2024 holding-only selection also retains 48h.

**The joint-grid choice is 12h EWMA / 1.5z / 72h maximum hold, with aligned
constituents and retained weights.** It scores 1.220 net Sharpe in 2024–2025
(0.668 with next-quote execution), compared with 0.991/0.530 for the original
strategy. Its 2017–2025 Sharpe is 1.652 versus 1.369. Retained sizing keeps the
original basket weights of selected crosses and drops the others, without
renormalising or rescaling to match the full basket's estimated risk.

Most of this difference is associated with the faster reference and sizing
choice: the same 12h/retained strategy already scores 1.180 with a 48h cap.
Extending that cap to 72h adds only 0.039 Sharpe in evaluation. In this period
there was one 48h time exit; the 72h version's longest trade lasted 65.87
market hours, so 72h and unlimited give identical evaluation trades. Across
the entire history, 72h and unlimited differ slightly. The grid's exact 72h
optimum should not be interpreted as well-established precision.

The highest evaluation Sharpe is 1.327 using 12h/1.5z and **normalized** aligned
weights, with caps of 72h or more tied. That is a retrospective evaluation
winner rather than the primary pre-2024 choice, and its realised volatility
is 6.51% versus 3.25% for retained sizing. Even the selected retained candidate
remains execution-sensitive: adding 0.25 bp per leg traded reduces its Sharpe
to 0.526, versus 0.570 for the original strategy. Its paired Sharpe improvement
interval spans zero. This is a candidate for further testing, not a demonstrated
robust replacement. The 2021 result remains negative.

This study changes the time exit and re-evaluates the equilibrium strategy.
It covers **324 portfolios**: EWMA half-lives 12/24/48/72 market hours;
entry thresholds 1/1.5/2; holding caps 48/72/120/240/480 market hours or unlimited,
plus the original 2×half-life cap when absent from that list; and aligned
constituents with risk-matched/retained/normalized sizing or a full-index control.

Every configuration regenerates its entire causal trade path. Extending a
position can block subsequent entries; this is not a retrospective extension
of only the losing trades. Selections and sizes stay fixed until the index
exits. The near/cross-mean exit remains 0.25z. Quotes are one-minute data,
2016–2025, with recorded spreads, gap PnL and final liquidation. **Financing
remains excluded**, including on extended holdings. PnL is the existing
additive log-return notional convention, not compounded cash-account returns.

The ranking is the minimum net Sharpe across 2017–2022, 2021–2022 and 2023,
requiring positive scores, at least ten completed validation trades and forty
active validation days. Choices are frozen before evaluation rankings are
computed. Unlike the earlier subset study, this joint grid compares full and
subset strategies directly, without requiring improvement over a parent in
every selection period. All 324 configurations are disclosed. The 2024–2025
sample was already examined in earlier research and is not a fresh holdout.

![Holding-time sensitivity and selected strategies](sgd_neer_results/holding_time/holding_time_comparison.png)

## Change only the current strategy's holding cap

24h equilibrium, 1.5z entry, aligned constituents, risk-matched sizing.
All Sharpes use daily marked-to-market PnL and sqrt(252) annualisation.

| max_hold_hours | development | validation | weak_2019_2021 | evaluation | evaluation_delay1 | all_2017_2025 |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | 1.368 | 2.324 | -0.197 | 0.991 | 0.530 | 1.369 |
| 72 | 1.425 | 2.257 | -0.073 | 0.764 | 0.325 | 1.356 |
| 120 | 1.464 | 2.345 | -0.081 | 0.711 | 0.273 | 1.379 |
| 240 | 1.456 | 2.345 | -0.100 | 0.728 | 0.289 | 1.377 |
| 480 | 1.456 | 2.345 | -0.100 | 0.728 | 0.289 | 1.377 |
| unlimited | 1.456 | 2.345 | -0.100 | 0.728 | 0.289 | 1.377 |

Actual holdings and cost/risk outcomes in 2024–2025:

| max_hold_hours | evaluation_trades | evaluation_mean_hold_h | evaluation_max_hold_h | evaluation_annual_cost_pct | evaluation_drawdown_pct |
| --- | --- | --- | --- | --- | --- |
| 48 | 189 | 18.581 | 48.000 | 4.473 | -3.004 |
| 72 | 181 | 19.472 | 72.000 | 4.412 | -3.080 |
| 120 | 180 | 19.623 | 120.000 | 4.379 | -3.007 |
| 240 | 178 | 19.843 | 137.383 | 4.357 | -2.960 |
| 480 | 178 | 19.843 | 137.383 | 4.357 | -2.960 |
| unlimited | 178 | 19.843 | 137.383 | 4.357 | -2.960 |

Holding statistics assign complete episodes to their entry year; daily PnL
and completed-trade counts follow calendar trading dates. A cap counts
scheduled market hours, with missing minutes included and weekend closure
excluded. A time exit without quotes fills at the next available quote.
Unlimited disables the time exit; the mean exit and sample-end liquidation
remain. Identical results at large caps indicate that those caps never bind
on this history, not independent evidence for a precisely optimal duration.

## Joint selection and retrospective comparison

| choice | half_life_h | entry_z | max_hold_h | constituents | sizing | selection_score | evaluation | evaluation_delay1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original strategy | 24 | 1.500 | 48 | aligned | risk_matched | 1.359 | 0.991 | 0.530 |
| Holding-only selection | 24 | 1.500 | 48 | aligned | risk_matched | 1.359 | 0.991 | 0.530 |
| Joint-grid selection | 12 | 1.500 | 72 | aligned | retained | 1.645 | 1.220 | 0.668 |
| Aligned/risk-matched selection | 12 | 1.500 | 72 | aligned | risk_matched | 1.554 | 1.089 | 0.476 |
| Best with 48h cap | 12 | 1.500 | 48 | aligned | retained | 1.617 | 1.180 | 0.631 |
| Highest evaluation Sharpe (retrospective) | 12 | 1.500 | 72 | aligned | normalized | 1.575 | 1.327 | 0.933 |

The retrospective row maximises 2024–2025 Sharpe after seeing that period;
it is shown explicitly apart from the choices selected on earlier data.
Equivalent parameter settings can tie when their time limits never bind.
Normalized subset sizing keeps aggregate SGD exposure but can increase
estimated risk; risk matching controls estimated entry risk, not realised risk.

Risk and weak-period performance of these choices:

| choice | weak_2019_2021 | all_2017_2025 | evaluation_vol_pct | evaluation_drawdown_pct | evaluation_mean_hold_h | evaluation_max_hold_h |
| --- | --- | --- | --- | --- | --- | --- |
| Original strategy | -0.197 | 1.369 | 2.934 | -3.004 | 18.581 | 48.000 |
| Holding-only selection | -0.197 | 1.369 | 2.934 | -3.004 | 18.581 | 48.000 |
| Joint-grid selection | 0.341 | 1.652 | 3.247 | -2.010 | 10.343 | 65.867 |
| Aligned/risk-matched selection | 0.223 | 1.511 | 2.920 | -1.945 | 10.343 | 65.867 |
| Best with 48h cap | 0.289 | 1.615 | 3.236 | -2.300 | 10.302 | 48.000 |
| Highest evaluation Sharpe (retrospective) | 0.167 | 1.542 | 6.510 | -2.335 | 10.343 | 65.867 |

Controlled comparisons separating the faster reference, holding cap and sizing:

| horizon_hours | entry_z | holding_hours | sizing | weak_2019_2021 | evaluation | all_2017_2025 | evaluation_delay1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 1.500 | 48 | risk_matched | -0.197 | 0.991 | 1.369 | 0.530 |
| 12 | 1.500 | 48 | risk_matched | 0.137 | 1.049 | 1.474 | 0.437 |
| 12 | 1.500 | 72 | risk_matched | 0.223 | 1.089 | 1.511 | 0.476 |
| 12 | 1.500 | 48 | retained | 0.289 | 1.180 | 1.615 | 0.631 |
| 12 | 1.500 | 72 | retained | 0.341 | 1.220 | 1.652 | 0.668 |

## Re-optimise other parameters at each holding cap

Each row below is chosen using only the pre-2024 ranking, then evaluated.

| max_hold_h | half_life_h | entry_z | constituents | sizing | score | evaluation | evaluation_delay1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48 | 12 | 1.500 | aligned | retained | 1.617 | 1.180 | 0.631 |
| 72 | 12 | 1.500 | aligned | retained | 1.645 | 1.220 | 0.668 |
| 120 | 12 | 1.500 | aligned | retained | 1.643 | 1.220 | 0.668 |
| 240 | 12 | 1.500 | aligned | retained | 1.643 | 1.220 | 0.668 |
| 480 | 12 | 1.500 | aligned | retained | 1.643 | 1.220 | 0.668 |
| unlimited | 12 | 1.500 | aligned | retained | 1.643 | 1.220 | 0.668 |

## Uncertainty and execution costs

Conditional 95% Sharpe intervals and paired differences from the original
24h/1.5z/48h strategy use 2,000 shared 20-day block resamples. They do not
adjust for the expanded strategy search or the previously inspected dates.

| policy | low | high | delta_low | delta_high |
| --- | --- | --- | --- | --- |
| equilibrium_12h_z1.5_hold48h__aligned__retained | 0.123 | 2.235 | -0.893 | 1.419 |
| equilibrium_12h_z1.5_hold72h__aligned__risk_matched | 0.045 | 2.172 | -0.898 | 1.227 |
| equilibrium_12h_z1.5_hold72h__aligned__retained | 0.173 | 2.259 | -0.843 | 1.427 |
| equilibrium_24h_z1.5_hold48h__aligned__risk_matched | -0.384 | 2.257 | 0.000 | 0.000 |

`stress.csv` contains next-quote execution, 1.5×/2× spread costs and an extra
0.25 bp per quoted leg traded. 2023 common-quote coverage is incomplete, and
the source cache does not preserve tick-age information. These limitations
remain material for both selection and evaluation.

Same-quote evaluation Sharpe under cost stresses:

| policy | base | extra_0.25bp_per_leg | spread_x1.5 | spread_x2 |
| --- | --- | --- | --- | --- |
| equilibrium_12h_z1.5_hold48h__aligned__retained | 1.180 | 0.483 | -0.033 | -1.203 |
| equilibrium_12h_z1.5_hold72h__aligned__normalized | 1.327 | 0.843 | 0.463 | -0.403 |
| equilibrium_12h_z1.5_hold72h__aligned__retained | 1.220 | 0.526 | 0.008 | -1.158 |
| equilibrium_12h_z1.5_hold72h__aligned__risk_matched | 1.089 | 0.360 | -0.186 | -1.393 |
| equilibrium_24h_z1.5_hold48h__aligned__risk_matched | 0.991 | 0.570 | 0.227 | -0.519 |

## Use the parameters in the strategy constructor

Copy a configuration from
[selected_hyperparameters.json](sgd_neer_results/holding_time/selected_hyperparameters.json)
into `SGDNEEREquilibriumStrategy(..., hyper_param_dict=params)`. Other
constructor defaults remain unchanged. In particular:

```python
# None still means the original default of 2 × equilibrium_half_life.
params["max_holding_period"] = pd.Timedelta(hours=120)  # Example: 5 market days.
# Or disable only the time exit:
params["max_holding_period"] = "unlimited"
```

The constructor now supports the explicit `"unlimited"` value; its default
has not changed. The research and strategy implementations are checked for
matching targets with default, extended and unlimited holdings, including
causal-prefix and missing-quote behavior. All four original 24h/1.5z/48h
portfolio variants reproduce the saved daily results at both execution delays.
All 72 relevant automated checks pass. Full-history constructor verification
also reproduces the selected retained and risk-matched portfolios and the
original baseline over 3,614,036 common minute quotes; maximum target error is
below 9e-16. Details are in `strategy_verification.json`.

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_holding.py
poetry run python src/systematic_trading/research/sgd_neer_codex/build_sgd_neer_holding_report.py
```

The [results folder](sgd_neer_results/holding_time/) includes the predeclared
protocol, frozen selections, every configuration's daily PnL, fill records,
entry allocations and sparse targets, annual scores, holding statistics,
execution stresses, input/source hashes and baseline verification.
