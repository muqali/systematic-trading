# Why equilibrium / aligned constituents struggled in 2019–2021

The saved default backtest points to weaker and less reliable reversals, larger
losses when reversals failed, and spreads consuming the remaining gross edge.
The weakness also appears in the full-index control. Aligned selection added a
component-selection problem, but did not create the overall weak period.

The dates matter: **2019–2021 were weak; 2022 was already a strong recovery**.
The later period is also uneven: 2024 was much weaker than 2022, 2023 or 2025.

This diagnosis uses the frozen 24-market-hour EWMA / 1.5z strategy, aligned
constituents, risk-matched sizing, one-minute observations and same-quote fills.
Recorded spreads are deducted; financing remains excluded. No parameters,
signals or historical backtest results were changed.

![Performance, execution sensitivity and subsequent index moves](sgd_neer_results/regime_diagnostics/regime_diagnostics.png)

## What actually changed

| Period | Net Sharpe | Completed episodes | Net winning trades | 48h timeouts | Mean gross index fade after 6h |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2016–2018 | 2.80 | 257 | 75.5% | 9.3% | 3.80 bp |
| 2019–2021 | -0.20 | 227 | 65.2% | 17.2% | 0.26 bp |
| 2022–2025 | 1.89 | 386 | 69.4% | 9.3% | 2.72 bp |

Sharpe uses daily marked-to-market PnL and sqrt(252) annualisation. Trade
statistics assign a whole episode to its entry year. The six-hour figure is
the signed, unscaled index return following the existing entry, before costs;
it can extend beyond the strategy's actual exit. It measures subsequent price
behavior independently of trade sizing. Horizons count scheduled market hours.

| Year | Gross Sharpe | Net Sharpe | Gross PnL, pp | Spread cost, pp | Net PnL, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2016 | 2.49 | 2.09 | 6.71 | 1.26 | 5.45 |
| 2017 | 3.41 | 2.85 | 9.89 | 1.91 | 7.98 |
| 2018 | 4.20 | 3.35 | 13.94 | 3.11 | 10.82 |
| 2019 | 1.19 | 0.48 | 3.99 | 2.38 | 1.61 |
| 2020 | 0.24 | -0.58 | 0.92 | 3.09 | -2.17 |
| 2021 | 0.46 | -0.42 | 1.72 | 3.29 | -1.58 |
| 2022 | 4.45 | 3.07 | 18.27 | 6.26 | 12.01 |
| 2023 | 3.31 | 2.32 | 9.43 | 3.37 | 6.06 |
| 2024 | 2.50 | 0.46 | 7.63 | 6.23 | 1.40 |
| 2025 | 2.38 | 1.51 | 7.62 | 3.02 | 4.61 |

PnL uses the existing additive log-return notional convention, expressed in
percentage points; these are not compounded cash-account returns. The 2016
daily series includes the strategy's initial warm-up.

## Weaker reversals and a larger failure tail

Compared with 2016–2018, average absolute entry deviation barely changed:
21.87 versus 21.02 bp. Average prior estimated 24-hour index volatility at
entry was also similar: 16.47 versus 16.51 bp. Lower average entry volatility
or smaller initial deviations therefore do not explain the weakness on their
own; those averages can still conceal differences within each period.

The subsequent price path did change. The mean signed index return six market
hours after entry fell from 3.80 to 0.26 bp. At 24 hours it fell from 7.00 to
2.15 bp, with the fraction of positive index moves falling from 70.2% to 52.2%.
In the later period the early reversal recovered, but the longer-horizon
markouts remained below the early sample: the 2022–2025 mean at 24 hours was
2.63 bp. A single unconditional claim that all mean reversion recovered would
therefore be too broad.

Under the actual exit rules, average holding time increased from 18.5 to
23.4 market hours. The timeout fraction rose from 9.3% to 17.2%. Average net
winning-trade PnL declined from 20.97 to 18.77 bp, while the average net losing
trade worsened from -26.36 to -37.47 bp at the configured position sizes.

The 39 timeout episodes entered during 2019–2021 lost 21.03 percentage points
of PnL in aggregate. Other episodes earned 19.21 points. This is a descriptive
decomposition, not evidence that timeouts can be identified in advance or
removed without losing profitable trades.

The weak years were not only a March 2020 shock. Negative quarters included
2019 Q3, 2020 Q3/Q4 and 2021 Q3/Q4. Both long-SGD and short-SGD episode cohorts
lost money over 2019–2021.

## Spreads consumed a much smaller gross edge

The period's gross daily Sharpe fell from 3.41 to 0.60, so execution costs
alone cannot explain the deterioration. But their effect became decisive:
2020 gross PnL of 0.92 points could not cover 3.09 points of spreads, and 2021
gross PnL of 1.72 points could not cover 3.29 points.

The mean realised round-trip cost per unit of aggregate selected SGD exposure
rose from 1.56 to 2.39 bp. Average exit full-index spreads rose from 1.62 to
2.93 bp. An entry cost hurdle does not constrain the spread at a mandatory
exit, and the assumed half-deviation reversion is a heuristic rather than an
estimated conditional payoff.

This is not simply a story of spreads becoming permanently worse: the later
period's mean round-trip unit cost was higher again, at 3.09 bp, but stronger
realised trade returns covered it. The 2024 result illustrates the distinction:
7.63 points gross profit became only 1.40 points net after 6.23 points of costs.

## Aligned contributors were not consistently the best trades

The full-index control's net Sharpe was **-0.13** in 2019–2021, versus **-0.20**
for aligned/risk-matched. Thus the broad weakness survives removal of the
subset-selection mechanism. It also survives a one-quote delay: the aligned
strategy's period Sharpes become 2.48, -0.40 and 1.49, respectively.

At the common index exit, equal-weighted per-unit synthetic-cross outcomes
show that the alignment criterion's usefulness changed:

| Period | Aligned cross mean gross | Aligned mean net | Opposing mean net |
| --- | ---: | ---: | ---: |
| 2016–2018 | 7.50 bp | 5.74 bp | 7.48 bp |
| 2019–2021 | 3.70 bp | 1.08 bp | 2.28 bp |
| 2022–2025 | 8.28 bp | 4.81 bp | -2.12 bp |

These cross observations share the same index trade episodes and are not
independent samples. They are also not the risk-weighted portfolio returns.
They show why identifying the contributor to an index deviation does not
establish which currency will reverse. The index can recover through a
previously opposing constituent while the original contributor continues moving.

SGD/JPY was the largest synthetic-cross drag in 2019–2021. It was selected in
210 of 227 episodes and contributed **-3.65 PnL percentage points**, including
its share of the USDSGD hedge and both legs' costs. SGD/USD contributed -0.68
points; EUR, AUD and GBP contributed +1.53, +0.42 and +0.55 points. These
contributions use entry-year cohorts, whose total differs from calendar PnL
because some trades cross year boundaries. They do not justify excluding JPY
based on hindsight.

## Economic interpretation: a moving reference can resemble mispricing

The model's “equilibrium” is a lagged EWMA, not an observed MAS band midpoint.
MAS allows its trade-weighted index to fluctuate within a band around a policy
path and can change the band's slope or level. That framework does not imply
that a deviation from a 24-hour EWMA must reverse within two days.
[MAS framework](https://www.mas.gov.sg/monetary-policy/singapores-monetary-policy-framework/faqs/section-2).

A persistent repricing can consequently trigger repeated fades while the EWMA
is still catching up. Even a near-mean exit need not mean that price returned
to its entry-time equilibrium. In 2019–2021, **14.4% of non-timeout exits** had
an adverse index price move since entry, versus **6.0%** in 2016–2018. Movement
of the reference, and changes in the volatility scale used by the z-score,
can bring the signal back inside its exit threshold without a profitable
price reversal.

Contemporaneous events make this a plausible economic mechanism:

- The largest losing episode in the weak cohort entered long SGD on
  1 August 2019 and timed out on 5 August, losing 2.24 PnL percentage points.
  BIS documents the tariff escalation and flight to safety starting on
  1 August, alongside renminbi weakness and spillovers to emerging-market
  currencies. The dates are consistent with fading a sustained repricing;
  this event match is not a causal identification exercise.
  [BIS, September 2019](https://www.bis.org/publications/markets-swing-trade-and-monetary-policy).
- MAS reduced the appreciation slope in October 2019. In March 2020 it
  adopted a zero-slope band at the prevailing index level, after describing
  depreciation from late January amid a worsening growth outlook. Fading the
  old moving average can conflict with an economic adjustment that policy
  accommodates.
  [October 2019 statement](https://www.mas.gov.sg/news/monetary-policy-statements/2019/mas-monetary-policy-statement-14oct19),
  [March 2020 statement](https://www.mas.gov.sg/news/monetary-policy-statements/2020/mas-monetary-policy-statement-30mar20).
- MAS resumed a positive slope in October 2021 and subsequently tightened
  further, including upward re-centring in October 2022. The strategy's strong
  2022 results mean that policy changes or trending policy bands, by
  themselves, are not sufficient explanations of poor performance.
  [October 2021 statement](https://www.mas.gov.sg/news/monetary-policy-statements/2021/mas-monetary-policy-statement-14oct21),
  [October 2022 statement](https://www.mas.gov.sg/news/monetary-policy-statements/2022/mas-monetary-policy-statement-14oct22).

There is also proxy risk: this research uses five currencies with fixed,
renormalised assumed weights, representing about 49.2% of the weight mass in
the project's larger assumed basket. MAS manages its own trade-weighted
basket. Neither the five-currency proxy nor an aligned subset is guaranteed
to inherit the same stabilisation properties in every cross-currency regime.

## Limits and useful next tests

The analysis explains the accounting and documents weaker reversals. It does
not establish a forecasting rule for when those conditions will recur. The
strategy and date ranges have already been inspected. Conditional 20-day-block
bootstrap 95% intervals for net Sharpe are [1.70, 3.98], [-1.31, 0.93], and
[0.95, 2.80], respectively; they do not adjust for strategy or period selection.
Common-quote coverage is unusually incomplete in 2023, and cached quote age
cannot be verified from these resampled inputs.

The most directly motivated next experiments would distinguish persistent
drift from temporary deviations using only prior data, estimate reversion
payoffs relative to realistic entry and exit costs, and compare index-based
exits with exits responsive to the selected portfolio. Loss limits or shorter
holding rules should be tested across all periods, including lost recoveries.
Any rule should be fixed on earlier data and evaluated chronologically;
excluding the bad calendar years or deleting JPY after seeing these outcomes
would not demonstrate an improvement.

## Reproduction and verification

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_regime_diagnostics.py
```

The runner reads the existing minute basket and saved trade/fill records.
It reconstructs all 870 complete episodes and verifies that gross PnL, costs
and net PnL reconcile to the saved daily backtest. Both quoted-leg and synthetic
SGD-cross attribution reconcile to the same totals. Entry deviations reproduce
the original decisions. No strategy code or saved source results are modified.

[Diagnostic tables and episodes](sgd_neer_results/regime_diagnostics/)
include annual and quarterly PnL, holding/exit statistics, side and currency
attribution, fixed-horizon index markouts, and Sharpe intervals.
