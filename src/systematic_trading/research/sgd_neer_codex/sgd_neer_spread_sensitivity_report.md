# 20% tighter spreads: notebook 24h equilibrium with aligned constituents

The notebook's **24h / 1.5z / aligned / risk_matched** strategy improves from
**1.306 to 1.522 net Sharpe** when its trades are held fixed and every bid/ask
spread is reduced by 20%. Gross PnL is unchanged. Regenerating signals with the
tighter quotes produces **1.518 net Sharpe**; the lower cost hurdle changes only
eight entry timestamps out of 804 entries.

This experiment uses the settings in the bottom equilibrium block of
[sgd_neer.ipynb](../sgd_neer.ipynb), inspected on 2026-09-27. The equilibrium
half-life remains 24 market hours, entry is at 1.5z, exit at 0.25z, and the default
maximum holding time remains 48 market hours. There is no parameter search.

## Same-quote execution: 2017–2025

| Scenario | Gross Sharpe | Net Sharpe | Gross cumulative PnL | Spread cost | Net cumulative PnL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original spreads | 2.338 | 1.306 | 71.358% | 32.776% | 38.582% |
| 20% tighter, original trades | 2.338 | 1.522 | 71.358% | 26.221% | 45.138% |
| 20% tighter, regenerate signals | 2.338 | 1.518 | 71.327% | 26.320% | 45.007% |

With trades fixed, cumulative spread cost falls by **6.555 percentage points**,
which increases cumulative net PnL by the same amount. Net annualized PnL rises
from 4.151% to 4.857%; net annualized volatility changes from 3.178% to 3.192%.
Net Sharpe therefore needs to be recalculated from the daily PnL series rather
than scaled by a constant.

## Same-quote execution: 2024–2025

| Scenario | Gross Sharpe | Net Sharpe | Gross cumulative PnL | Spread cost | Net cumulative PnL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original spreads | 2.439 | 0.991 | 15.257% | 9.247% | 6.010% |
| 20% tighter, original trades | 2.439 | 1.294 | 15.257% | 7.398% | 7.860% |
| 20% tighter, regenerate signals | 2.458 | 1.304 | 15.375% | 7.459% | 7.916% |

All three cases have 189 entries in these two years. The cost-only change adds
**1.849 percentage points**, or 30.8%, to cumulative net PnL. These dates have
already been inspected in prior research and are not an untouched holdout.

## Why regenerated signals differ slightly

The spread transformation is applied to all five quoted legs at every timestamp:

```python
bid_new = mid + 0.8 * (bid - mid)
ask_new = mid + 0.8 * (ask - mid)
```

The mid, index level, EWMA reference, z-score, component deviations and volatility
estimates remain identical. However, entry also requires:

```python
0.5 * abs(index_deviation) > 1.5 * full_index_spread
```

Reducing the spread makes that condition easier to satisfy. Regeneration retains
804 total entries but moves eight of them earlier; 796 entry timestamps remain
identical. Every target at a shared target timestamp is also unchanged. Earlier
entries can have different component allocations and sizes, based on information
at the new entry time, so both gross PnL and realized costs can change.

The fixed-trade scenario isolates the transaction-cost benefit. Regeneration
measures the complete strategy's response to changed quotes. Its 2017–2025 gross
PnL is slightly lower, while its 2024–2025 gross PnL is slightly higher.

## One-common-quote delay sensitivity

The notebook uses delay zero. Keeping all settings fixed but delaying execution
by one common valid quote gives:

| Period | Original net Sharpe | Tighter, original trades | Tighter, regenerate signals |
| --- | ---: | ---: | ---: |
| 2017–2025 | 1.014 | 1.230 | 1.209 |
| 2024–2025 | 0.530 | 0.837 | 0.818 |

The delay is normally one minute; it is longer across missing quotes or market
closures. It applies to both entries and exits. Lower spreads improve these
results too, while execution timing remains consequential.

## Data, accounting and verification

- One-minute quotes for USDSGD, EURUSD, AUDUSD, GBPUSD and USDJPY; 3,243,766
  common valid market quotes from 2017-01-01 inclusive to 2025-12-31 17:00 New
  York exclusive. Indicators start in 2017, matching the current notebook.
  Previous full-history research initialized indicators in 2016, explaining its
  different 2017–2025 baseline.
- Sharpe uses daily portfolio PnL, annualized with sqrt(252), retaining flat
  observed weekdays. Trading dates roll at 17:00 New York. There are 2,342
  observed dates in the full sample and 521 in 2024–2025. Existing data gaps,
  including the 2023 outage, are preserved.
- Cumulative PnL is the sum of weighted pair log-return PnL under the repository's
  existing notional convention; percentages are additive, not compounded wealth
  returns. All legs are summed. Entries, exits and final liquidation are costed.
  Forward points and financing remain excluded as requested.
- Exact directional log execution costs are recomputed at the hypothetical
  bid/ask prices. Fixed-trade total cost is 0.79999947 times baseline, effectively
  80%; the tiny difference from exact linear scaling comes from the log formula.
- The baseline reproduces the notebook's stored gross/net Sharpes and total
  spread cost to its displayed six decimal places. Baseline and regenerated
  daily gross/net PnL agree with `AggressiveTrader` under both execution delays
  to a maximum absolute difference below 6e-16.
- Assertions verify identical fixed-trade positions and gross PnL, the daily
  cost-to-net-PnL identity, unchanged mids, and unchanged non-spread features.
  Original quote files, the notebook and the strategy implementation are not
  modified by this experiment.

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_spread_sensitivity.py
```

The runner requires the existing `tmp/sgd_neer_m1_research/data.pkl` cache.
[comparison.csv](sgd_neer_results/spread_sensitivity/comparison.csv) includes
annual breakdowns; [the results folder](sgd_neer_results/spread_sensitivity/)
also contains daily PnL, trade ledgers, entry allocations and target positions.
[verification.json](sgd_neer_results/spread_sensitivity/verification.json)
records parameters, input/source hashes and numerical checks.
