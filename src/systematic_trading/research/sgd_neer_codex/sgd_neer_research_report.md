**Execution update:** [The same-bar follow-up](sgd_neer_same_bar_report.md) reruns all 34 configurations with immediate fills and aligned forecast targets. The corrected baseline improves from −2.29 to −0.52 net Sharpe in 2024–2025. The original one-bar-delay study follows below.

The experiments do **not** support a profitable, robust mean reversion strategy on the current five-pair proxy at the supplied spreads and a one-bar execution delay. Correcting the portfolio is necessary, but it exposes a much larger cost burden. I tested 34 individual configurations and a frozen ensemble; none of the adequately active new configurations had positive net Sharpe in both development and validation.

The companion [notebook](sgd_neer_cost_aware.ipynb) contains the tables and charts. [All results and reproduction instructions](sgd_neer_results/README.md) are saved locally. The original exploratory notebook was left intact.

**The most consequential corrections**

The index in `build_rets_vs_sgd` uses weights normalised over available currencies. For the notebook's five pairs, its return is:

```text
index = -1.000000 × USDSGD return
        -0.285741 × EURUSD return
        -0.123574 × AUDUSD return
        -0.034601 × GBPUSD return
        +0.178327 × USDJPY return
```

Every SGD cross contains the same short USDSGD leg. The old `_index_signal_to_pair_signal` overwrote that leg each time through the currency loop. With the current currency order, it ended at `-0.0182 × signal`, rather than `-1 × signal`. Other legs used unnormalised weights. This produced a different portfolio from the one whose mean reversion was measured. Consequently, the apparent performance of the old implementation cannot be interpreted as the performance of the specified index.

I corrected the pair mapping and spread calculation to use the same normalisation as the index. Custom weights now also reach the return calculation. Calibration now checks each forward label's actual ending timestamp: simply subtracting a wall-clock horizon is insufficient when observation-based labels cross a weekend or missing quotes.

The new backtest aligns valid quotes, fills at the next observed quote, earns returns only after the fill, and charges every entry, adjustment and exit, including final liquidation. It marks held positions across missing observations rather than deleting those returns. It computes Sharpe from New York trading-day PnL, using 252 days and excluding weekends, rather than scaling the standard deviation of serially dependent 15-minute PnL. All controls are re-evaluated under this common execution engine. These conventions differ from the exploratory notebook, so its reported ~0.5 Sharpe is not an apples-to-apples benchmark.

The repository's log-return/notional PnL convention is retained. This is not a cash-accounting spot-FX simulator with settlement, conversion balances, financing and market impact.

**What the data says**

There are 141,615 common valid quotes out of 148,701 union timestamps for 2020–2025. In 2023 only 18,408 of 24,752 timestamps have all five quotes: about 25.6% are incomplete. These are numerous missing intervals, particularly February–July, not a single missing quarter. Signals reset after gaps; long-history models consequently have fewer opportunities. No quotes are forward-filled in the new backtest. Cached bars lack the underlying quote timestamps, so simultaneous freshness across all legs remains unverified.

The five available currencies account for 49.2% of the total weight in `DEFAULT_WEIGHTS`, then get renormalised to 100%. CNH and most regional Asian constituents are absent. These are also assumed, fixed weights, not an observed official MAS basket.

MAS manages a basket along a potentially appreciating policy band; it can change the slope, centre and width. This does not imply reversion to a fixed level or a guaranteed 15-minute arbitrage. The official framework explicitly allows the S$NEER to float within the band. [MAS framework](https://www.mas.gov.sg/monetary-policy/singapores-monetary-policy-framework/faqs/section-2). The basket itself is undisclosed. [IMF discussion](https://www.elibrary.imf.org/view/journals/002/2022/233/article-A001-en.xml).

On continuous development-period observations, the descriptive variance ratio is about **0.70 at two hours** and **0.54 at twenty hours**, relative to 15 minutes. There is measurable return reversal. However, a variance ratio below one is not a profitability test or proof that MAS intervention caused the reversal; changing volatility, quote staleness and the proxy construction can also affect it.

For the notebook's two-hour past / thirty-minute future return regression, development-period coefficients are:

| Execution delay before measuring future returns | Beta | HAC t-statistic | R² |
|---|---:|---:|---:|
| None | −0.0480 | −10.93 | 0.75% |
| 15 minutes | −0.0243 | −5.94 | 0.19% |
| 30 minutes | −0.0165 | −4.23 | 0.09% |

Thus a hypothetical 10 bp index move predicts only about **0.24 bp** of reversal over the next thirty minutes after a 15-minute delay. This is an illustration from a linear fit, not a tail-conditional forecast. The median basket round-trip spread is **1.71 bp in development** and **1.57 bp across the full sample**. Statistical significance survives the delay, but much of the economically usable signal does not.

**Experiments and selection**

2020 provides warm-up history. All policies are compared on 2021–2022 development performance, then 2023 validation performance. Forecast models refit weekly on trailing data, using only labels realised before the fit, with causal volatility scaling and ridge shrinkage. Their target begins after the one-bar execution delay. The final shortlist was frozen before computing 2024–2025 performance.

The 34 individual configurations were:

| Family | Configurations | Main idea |
|---|---:|---|
| Return reversal | 6 | Fade 2h/8h/24h moves, entry 1.5/2 standard deviations |
| Moving-equilibrium deviation | 6 | Fade deviations from an EWMA with 4h/16h/64h half-lives |
| Forecast plus hysteresis | 6 | Forecast 1h/4h/12h returns, using 12/52 weeks of training |
| Forecast plus no-trade region | 12 | Same horizons/training windows, two cost buffers |
| Component execution | 4 | Predict individual legs from basket features; trade all five or only EURUSD/USDJPY |

The last two families were added after observing development results, before inspecting final evaluation performance. This is an adaptive research search, not a preregistered statistical test. All attempted configurations are retained. Policies require at least 40 validation days with trades, avoiding misleading Sharpes from one or two events. Ranking uses the lower of development and validation Sharpe. Every eligible new policy failed the requirement that both be positive, so the recorded decision was **reject**. The three best eligible new policies and their equal-weight, netted ensemble were carried into final evaluation for diagnosis only.

Simple rules use distinct entry and exit thresholds, hold size between entry and exit, impose a time limit and require a spread cushion. Forecast policies use multiple return horizons, a 5% ex-ante basket risk budget and a cap of three basket units. That risk budget is not a promised realised volatility: these strategies spend much of their time flat. New policies request liquidation before 17:00 New York; the evaluated candidates had no residual exposure at rollover in 2024–2025. The legacy controls do have rollover exposure, and financing is missing from their base results.

The most useful alternative implementation is the no-trade region. Given forecast `mu`, current position `q_prev`, positive risk curvature `A` and one-way cost `c`, maximise:

```text
q × mu − 0.5 × A × q² − c × abs(q − q_prev)
```

Keep the existing position if it lies between `(mu-c)/A` and `(mu+c)/A`; otherwise move to the nearest boundary, subject to the position cap. This puts the forecast and trading cost in consistent return units and avoids resetting to a noisy point target. The implementation is a one-period utility approximation, not a solution to the full dynamic optimisation problem. The broader connection between signal persistence and gradual trading is discussed by [Gârleanu and Pedersen](https://www.nber.org/papers/w15205).

**Frozen evaluation: 2024–2025**

All figures below use next-quote execution and observed bid/ask spreads.

| Policy | Development net SR | Validation net SR | Evaluation gross SR | Evaluation net SR |
|---|---:|---:|---:|---:|
| Original implementation, captured signal | 0.12 | −0.04 | 0.39 | −0.32 |
| Correct basket mapping, original signal | −2.07 | −3.13 | 2.16 | −2.83 |
| Corrected baseline, including cost threshold and label purge | −2.02 | −3.22 | 2.08 | −2.29 |
| 12h forecast, 52-week fit, wider no-trade region | −0.06 | −0.42 | 0.80 | −0.65 |
| 4h equilibrium, entry z=2 | −0.69 | 0.33 | −0.07 | −1.62 |
| 4h forecast, 52-week fit, narrower no-trade region | −0.63 | −0.95 | 0.38 | −1.11 |
| Frozen three-policy ensemble | −0.64 | −0.25 | 0.29 | −1.75 |

The strongest eligible alternative has net Sharpe −1.40 in 2024 and +0.26 in 2025. Its two-year Sharpe is −0.65, with a 20-trading-day block-bootstrap 95% interval of approximately **[−2.49, +1.01]**. The interval is descriptive and does not adjust for model selection or validate a positive edge. It traded on 88 evaluation days, realised about 0.41% annual volatility, and lost about 0.26% annually under the model's notional convention.

For that fixed alternative:

| Sensitivity | Evaluation net Sharpe |
|---|---:|
| Base observed spread cost, 15-minute delay | −0.65 |
| Half the spread cost | +0.09 |
| 1.5× spread cost | −1.34 |
| 2× spread cost | −1.95 |
| Extra 0.25 bp per traded leg, per side | −1.21 |
| Same-bar execution, optimistic diagnostic | +0.20 |
| Two-bar execution delay | −1.84 |

With positions held fixed, transaction costs would need to fall below about **56% of the observed spread cost** merely to achieve positive aggregate PnL for this alternative. This is not an optimised lower-cost execution strategy. No realistic cost stress establishes a compelling Sharpe improvement.

The final dates were excluded from this run's selection, but they had already been explored in the original notebook. They are therefore a held-out period for this experiment, not independently pristine evidence. No parameters were retuned after these results.

**Where to take the research next**

1. **Investigate execution before adding signal complexity.** Reconstruct decision-time and executable quotes at tick/one-minute resolution, check quote age on every leg, and measure how much reversal remains after achievable latency. A whole 15-minute delay may be conservative, but filling instantly at a historical bar close is not proof of executability. Passive execution needs an actual fill/adverse-selection model, not assumed spread capture.
2. **Improve the economic index.** Obtain synchronous CNH and regional-currency quotes, or a reliable historical S$NEER estimate. Test deviations from an estimated crawling equilibrium, with separate policy regimes and realistic carry. The current five-pair proxy is inadequate evidence for an official-band strategy.
3. **Use expected executable PnL to decide whether to trade.** Retain the new position/cost framework; estimate the distribution of post-latency returns and compare the expected benefit with entry, exit and financing costs. Test a small number of predeclared session/regime hypotheses on genuinely new data. These hypotheses were not tested after seeing the final evaluation.

For the current data and assumptions, remaining flat is preferable to selecting a negative-Sharpe configuration or promoting a sparse in-sample winner. There is evidence of reversal, but no demonstrated net edge to scale.

**Verification**

Eight new tests pass: basket replication and custom weights; invalid/missing constituents; execution timing; entry/exit/final costs and slippage; missing-quote PnL; trading-day aggregation; prefix invariance of forecasts/signals and label maturity; weekend-label leakage. The combined related test run had 14 passes and one pre-existing failure: `tests/test_aggressive_trader.py` passes `fwdptstn`, while the existing constructor accepts `fwdpts_tn_dict`. That unrelated API/test mismatch was left unchanged. The original notebook and the user's existing trendiness edits were preserved.
