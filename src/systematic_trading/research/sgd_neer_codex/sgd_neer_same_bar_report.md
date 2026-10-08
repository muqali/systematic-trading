**Same-bar execution follow-up**

A full 15-minute execution delay is too coarse to be the only primary assumption for a strategy calculated from tick-derived 15-minute quotes. The original study did test same-bar fills, but only as a sensitivity test of an already selected shortlist. In particular, the new forecast models remained trained to predict returns beginning one bar later. This follow-up makes zero-bar execution a primary case and retrains the forecasts accordingly.

The convention is causal at bar resolution: observe the quote at time t, calculate the signal, trade at that quote's bid/ask, and earn returns from t onward. It does **not** give the new position the return ending at t. In pandas terms, with `ret[t] = log(mid[t]/mid[t-1])`, gross PnL is still `signal.shift(1) * ret`; the extra shift of the signal into an execution position is removed. Spreads are charged on every adjustment and on final liquidation.

The aligned forecast label is now `level[t+h] - level[t]`, versus `level[t+1+h] - level[t+1]` in the earlier model. Training still uses only labels whose actual end timestamp precedes the calibration date. Both label conventions are available through `Policy.execution_delay` and `rolling_forecast(..., execution_delay=...)`; execution uses `backtest(..., delay=...)`.

**What had already been tested**

The earlier frozen candidate with a 12-hour forecast, 52-week training window and wider no-trade region achieved **+0.198 net Sharpe** in 2024–2025 when executed immediately, retaining its original delay-trained forecasts. The corrected baseline achieved **−0.519**. Those are valid sensitivity results, but the earlier +0.198 does not come from the newly retrained zero-delay model. Refitting changes the signal; it does not mechanically improve performance.

**Full same-bar rerun**

All 34 individual configurations were rerun with same-bar fills. Forecast and component models were retrained with zero-delay labels. The same 2021–2022 development / 2023 validation split and minimum of 40 validation trading days were retained. Selection used the lower development/validation net Sharpe, before calculating this rerun's 2024–2025 results. No thresholds or other hyperparameters were changed.

| Policy | Development net SR | Validation net SR | 2024–2025 gross SR | 2024–2025 net SR |
|---|---:|---:|---:|---:|
| Corrected baseline | −0.85 | −2.47 | 2.80 | −0.52 |
| 4h equilibrium, entry z=2 | +0.10 | +1.36 | 0.44 | −0.93 |
| Retrained 12h forecast, wider no-trade region | −0.27 | +0.23 | 0.65 | −0.45 |
| Retrained 4h forecast, wider no-trade region | −0.27 | +0.11 | 0.24 | −0.45 |
| Fixed three-policy ensemble | −0.08 | +0.95 | 0.62 | −0.92 |

The equilibrium rule was the only new configuration to have positive net Sharpe in both development and validation while meeting the activity requirement. Its subsequent evaluation was negative. The 12-hour forecast rule had −1.67 net Sharpe in 2024 and +0.59 in 2025, so the positive final year is not evidence of consistent performance. These evaluation dates were already seen in the earlier research; this follow-up is not an independent untouched test.

Immediate execution substantially helps the corrected baseline: net Sharpe improves from **−2.294 to −0.519**. At half the recorded spread cost, its same-bar net Sharpe is **+1.18**. That identifies transaction cost as a material constraint; it does not establish that half-spread-cost execution is achievable. Financing is still excluded from the baseline, which can hold across rollover. The shortlisted new rules were flat at rollover in the evaluation period.

The appropriate conclusion is narrower than “a 15-minute delay kills the strategy”: same-bar execution helps materially and should be a primary assumption here, but neither the earlier +0.20 sensitivity result nor the retrained shortlist demonstrates a strong, stable net edge at the supplied spreads.

**Data and verification**

This rerun uses the existing tick-derived 15-minute bid/ask caches. It is not a tick-by-tick fill replay. The configured raw-data directory `/Users/jinliu/Programming/data/tickquote` currently contains no tick CSVs, so actual next-tick latency and slippage were not measured. Tick data may exist in another location.

All 11 tests in `tests/test_sgd_neer_backtest.py` pass, including same-bar PnL timing, explicit immediate-return training labels, and prefix invariance for both zero- and one-bar forecast delays.

[Saved same-bar results](sgd_neer_results/delay_0/comparison_evaluation.csv) · [All development trials](sgd_neer_results/delay_0/development_leaderboard.csv) · [Frozen selection](sgd_neer_results/delay_0/frozen_selection.json)

From the repository root:

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_research.py --phase development --execution-delay 0
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_research.py --phase evaluation --execution-delay 0
```

Zero-delay results are written to `sgd_neer_results/delay_0/`; the earlier one-bar results remain separately available. The updated notebook has an appended same-bar comparison section.
