# SGD NEER: multi-day horizons and entry z-scores

Forward-point adjustments are **excluded**, as requested. All net Sharpes below deduct recorded bid/ask spreads; they are annualised from daily New York trading-date PnL using √252.

## What had already been tried

The original M15 research tested return horizons up to 24 hours and EWMA half-lives up to 64 hours, with entry thresholds 1.5 and 2. The minute study used 2.0 for its return-reversal and equilibrium rules, including the previously reported one-hour rule. Forecast no-trade bands were gated by spreads rather than a hard 2z entry test. Those new research policies requested daily liquidation before rollover, so long lookbacks did not amount to deliberate multi-day holding.

## New experiment

The declared grid contains **48 rules**: three signal types × four horizons (1/3/5/10 business days) × four entry thresholds (1.5/2/2.5/3). Signals update at **16:00 New York once per business day**; fills and PnL use the five-pair **one-minute quotes** for 2016–2025. Positions can remain open overnight and through weekends. This changes signal frequency, volatility estimation and holding rules as well as lookback length; it is not an isolated one-parameter horizon comparison with the intraday study.

- **Return reversal:** fade the H-day basket return after subtracting its prior 63-day-half-life EWM mean. Divide by its prior EWM standard deviation. A heuristic half-reversion move supplies the spread-entry hurdle.
- **Equilibrium:** fade the level's deviation from the previous day's H-day-half-life EWMA. Normalise by prior EWM dispersion of these daily deviations (63-day half life). This is an adaptive mean, not an estimated MAS policy-band midpoint.
- **Forecast:** regress the next H-day return on the past H-day return weekly, using the preceding 252 business dates and at least 100 mature labels. Clip beta at zero to retain mean-reversion forecasts; no intercept is traded. Standardise the forecast using its prior 126-day mean/std, requiring 60 valid prior predictions. This directly tests a z-score gate on the **predicted return**, rather than only on the observed price move.

Enter when |z|≥threshold and |edge|>1.5× the contemporaneous full basket spread. Exit near/crossing the mean (0.25z), on a forecast sign reversal/near-zero z, or after **2×H business days**. Size once on entry, targeting 5% annual volatility using prior daily returns, capped at three basket units. No pyramid or daily size rebalancing is used.

Business-date grids retain missing dates. Missing 16:00 quotes prevent new price-dependent decisions. A pre-known time limit can still request an exit; it fills at the next available common minute quote and retains intervening gap PnL. All legs, entry/exit costs and final liquidation are included. The sparse-trade accountant was checked against the general five-leg engine at zero-, one- and fifteen-minute delays.

## Findings

One-day signals are more promising than the previous intraday tests. The **one-day return-reversal rule at 1.5z** has a 2024–2025 net Sharpe of **0.729**, with 48 completed trades. At **2z** it scores **0.544**, with 21 trades. Stricter thresholds do not provide a monotonic improvement, and 2.5–3z leave very few trades.

The highest 2024–2025 result in the declared grid is the **one-day equilibrium rule at 1.5z**, at **1.286**. However, its 2017–2022 development Sharpe is only **0.058**, versus **0.901** for one-day return reversal at 1.5z. The largest final-period Sharpe is therefore an exploratory observation, not a model selected on past data.

The one-day forecast model at 2z scores **0.804** in development and **0.344** in evaluation; its 2025 Sharpe is **-0.897**. It does not dominate the simpler return-reversal rule.

| policy | entry_z | sharpe | completed_trades |
| --- | --- | --- | --- |
| return_1d_z1.5 | 1.500 | 0.729 | 48 |
| return_1d_z2 | 2.000 | 0.544 | 21 |
| return_1d_z2.5 | 2.500 | 0.826 | 10 |
| return_1d_z3 | 3.000 | 0.931 | 5 |

All declared horizons and thresholds: **2024–2025 net Sharpe, immediate execution**.

| kind | horizon_days | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- |
| equilibrium | 1 | 1.286 | 0.674 | 0.621 | 0.074 |
| equilibrium | 3 | 0.365 | -0.123 | 0.463 | -0.172 |
| equilibrium | 5 | -0.031 | 0.038 | 0.165 | 0.010 |
| equilibrium | 10 | -0.031 | -0.457 | -0.161 | -0.161 |
| forecast | 1 | -0.055 | 0.344 | -0.023 | -0.125 |
| forecast | 3 | -0.577 | -0.601 | -0.627 | -0.430 |
| forecast | 5 | -0.455 | -0.342 | -0.361 | -0.421 |
| forecast | 10 | 0.221 | 0.291 | -0.037 | -0.090 |
| return | 1 | 0.729 | 0.544 | 0.826 | 0.931 |
| return | 3 | 0.234 | -0.458 | -0.462 | -0.212 |
| return | 5 | -0.421 | -0.953 | -0.029 | -0.024 |
| return | 10 | 0.493 | 0.248 | 0.758 | 0.603 |

## Stability across periods

| policy | development | recent_development | validation | 2024 | 2025 | evaluation | all_2017_2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| return_1d_z1.5 | 0.901 | 0.929 | 0.392 | 1.263 | 0.252 | 0.729 | 0.811 |
| return_1d_z2 | 0.930 | 1.390 | 0.316 | 0.966 | 0.215 | 0.544 | 0.778 |
| return_1d_z2.5 | 0.396 | 1.031 | 1.273 | 0.799 | 0.858 | 0.826 | 0.498 |
| return_1d_z3 | 0.123 | 0.277 | 1.273 | 1.415 | 0.491 | 0.931 | 0.339 |
| equilibrium_1d_z1.5 | 0.058 | 0.244 | 0.794 | 1.940 | 0.658 | 1.286 | 0.380 |
| equilibrium_1d_z2 | 0.011 | 0.026 | 1.695 | 1.212 | 0.280 | 0.674 | 0.208 |
| forecast_1d_z2 | 0.804 | 1.031 | 1.273 | 1.257 | -0.897 | 0.344 | 0.664 |

The primary selection rule was fixed before final-period scoring: rank the minimum net Sharpe across 2017–2022, 2021–2022 and 2023, with at least 10 completed validation trades and 40 active validation days. **No rule met that activity floor**, so the frozen shortlist is empty and no ensemble was constructed. The positive numbers above remain diagnostic results rather than a validated winner. The complete grid is disclosed, including weak configurations.

This is materially affected by data coverage: only **148 of 260 (56.9%)** common 16:00 quotes exist in 2023. The rule at one-day return/2z completed only 2 validation trades. Missing decisions can change which excursions are traded. In 2024 and 2025, decision coverage is 96.6% and 99.2%, respectively. Cached minute quotes may themselves be forward filled by up to ten minutes; this study adds no quote filling.

## Execution and costs

At daily horizons the one-minute execution penalty is much smaller than in the intraday experiment. Delays count common valid quote rows; gaps can extend elapsed time.

| policy | 0 | 1 |
| --- | --- | --- |
| equilibrium_1d_z1.5 | 1.286 | 1.289 |
| equilibrium_1d_z2 | 0.674 | 0.650 |
| forecast_1d_z2 | 0.344 | 0.306 |
| return_1d_z1.5 | 0.729 | 0.687 |
| return_1d_z2 | 0.544 | 0.541 |
| return_1d_z2.5 | 0.826 | 0.805 |
| return_1d_z3 | 0.931 | 0.936 |

| policy | fill_delay_minutes | scenario | sharpe |
| --- | --- | --- | --- |
| return_1d_z1.5 | 0 | base | 0.729 |
| return_1d_z1.5 | 0 | spread_x1.5 | 0.460 |
| return_1d_z1.5 | 0 | spread_x2 | 0.190 |
| return_1d_z1.5 | 0 | extra_0.25bp_per_leg | 0.517 |
| return_1d_z1.5 | 1 | base | 0.687 |
| return_1d_z1.5 | 1 | spread_x1.5 | 0.402 |
| return_1d_z1.5 | 1 | spread_x2 | 0.118 |
| return_1d_z1.5 | 1 | extra_0.25bp_per_leg | 0.476 |
| return_1d_z1.5 | 15 | base | 0.496 |
| return_1d_z2 | 0 | base | 0.544 |
| return_1d_z2 | 0 | spread_x1.5 | 0.432 |
| return_1d_z2 | 0 | spread_x2 | 0.317 |
| return_1d_z2 | 0 | extra_0.25bp_per_leg | 0.433 |
| return_1d_z2 | 1 | base | 0.541 |
| return_1d_z2 | 1 | spread_x1.5 | 0.425 |
| return_1d_z2 | 1 | spread_x2 | 0.308 |
| return_1d_z2 | 1 | extra_0.25bp_per_leg | 0.430 |
| return_1d_z2 | 15 | base | 0.348 |
| equilibrium_1d_z1.5 | 0 | base | 1.286 |
| equilibrium_1d_z1.5 | 0 | spread_x1.5 | 1.086 |
| equilibrium_1d_z1.5 | 0 | spread_x2 | 0.882 |
| equilibrium_1d_z1.5 | 0 | extra_0.25bp_per_leg | 1.117 |
| equilibrium_1d_z1.5 | 1 | base | 1.289 |
| equilibrium_1d_z1.5 | 1 | spread_x1.5 | 1.087 |
| equilibrium_1d_z1.5 | 1 | spread_x2 | 0.880 |
| equilibrium_1d_z1.5 | 1 | extra_0.25bp_per_leg | 1.121 |
| equilibrium_1d_z1.5 | 15 | base | 1.138 |
| equilibrium_1d_z2 | 0 | base | 0.674 |
| equilibrium_1d_z2 | 0 | spread_x1.5 | 0.541 |
| equilibrium_1d_z2 | 0 | spread_x2 | 0.404 |
| equilibrium_1d_z2 | 0 | extra_0.25bp_per_leg | 0.588 |
| equilibrium_1d_z2 | 1 | base | 0.650 |
| equilibrium_1d_z2 | 1 | spread_x1.5 | 0.516 |
| equilibrium_1d_z2 | 1 | spread_x2 | 0.380 |
| equilibrium_1d_z2 | 1 | extra_0.25bp_per_leg | 0.563 |
| equilibrium_1d_z2 | 15 | base | 0.511 |

`bootstrap.csv` contains 20-day circular-block Sharpe intervals with 2,000 draws for every rule. Some sparse strategies produce bootstrap samples with zero volatility and undefined Sharpe: intervals are conditional on nonzero-volatility draws, and the excluded draw counts are disclosed. These intervals do not correct for searching 48 rules or previous experiments.

The 2024–2025 data have already been inspected in earlier work; this is **not an untouched holdout**. The 2016 history warms up the statistics, while forecast z-scores additionally require 60 previous forecasts. PnL uses the existing fixed log-return notional convention, rather than a full cash-settlement ledger. No forward points, financing, commission or market impact are applied. Extra slippage is shown as sensitivity only.

## Reproduce

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_multiday.py
poetry run pytest -q tests/test_sgd_neer_multiday.py tests/test_sgd_neer_backtest.py
```

The runner reuses the audited minute basket in `tmp/sgd_neer_m1_research/data.pkl`. If absent, run the minute runner's `--stage prepare` first. Policy definitions, source/input manifests, full daily PnL, actual fill ledgers, regression fits and all 48 results are saved under `sgd_neer_results/multiday`.

[Executed notebook](sgd_neer_multiday.ipynb) · [Minute-study report](sgd_neer_minute_report.md)
