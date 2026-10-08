# SGD NEER: one-minute follow-up

The corrected original regression strategy, recalculated every minute, has a **2024–2025 net Sharpe of -1.542 with immediate fills and -2.456 with a one-minute fill delay**. The newly selected equal-weight ensemble scores -0.354 and -1.209, respectively. These are after recorded bid/ask spreads, before financing.

The best 2024–2025 Sharpe among the evaluated minute models is the **one-hour equilibrium rule: 0.488**, with an approximate 95% block-bootstrap interval [-0.934, 2.170]. Its 2024 Sharpe is 2.200, but 2025 is -0.652; its 2017–2022 development Sharpe is -0.175. Adding just 0.25 bp per traded leg reduces its evaluation Sharpe to -1.131. This does **not establish a robust improvement** over the original research target.

That rule fades departures from a causal 60-minute-half-life EWMA: enter at |z|≥2 with an estimated reversal greater than 1.5× the full basket spread; exit near the mean (|z|≤0.25 on the position side) or after eight hours, with a pre-rollover exit request. Volatility sizing targets 5% annual risk before a three-basket-unit cap. Its theoretical sizing is approximate; the spread/latency stresses are the binding limitation here.

Using the same minute data but keeping the original strategy's signals on a 15-minute grid gives **-0.670** with immediate execution, **-0.881** with one-minute execution delay, and **-2.274** with 15-minute delay. Faster execution helps this control substantially; recalculating more frequently is a separate change and adds trading costs.

## Data and implementation

- `freq = pd.Timedelta(minutes=1)`; USDSGD, EURUSD, AUDUSD, GBPUSD and USDJPY; all 50 exported 2016–2025 files. SHA-256 input manifest saved.
- 3,614,036 common valid observations of 3,715,194 timestamps; median five-leg round-trip spread 1.459 bp. Common coverage in 2023 is only 77.45%, although each pair has about 88.5% coverage. Gaps differ across pairs.
- Cached quotes may have been forward filled for up to ten minutes. This research adds no filling before constructing the basket. Cached quotes have no original tick-age/size information, so immediate execution remains an assumption.
- At common quarter-hour timestamps, a comparison with the old M15 caches in 2021, 2024 and 2025 found 99.94–100% equal midquotes per pair/year. The new cache still changes missing-data coverage. Matched controls retain missing grid rows before the original strategy constructs returns, avoiding an inconsistent treatment of gap returns in its signal. Preliminary controls that compressed the grid first were superseded.
- Normalised tradable proxy exposure: USDSGD −1, EURUSD −0.285741, AUDUSD −0.123574, GBPUSD −0.034601, USDJPY +0.178327. This remains the five-pair proxy, not the full official MAS basket.
- Features retain horizons of 1, 4, 16 and 48 hours. Volatility retains a ten-trading-day half-life/history, a twenty-day trailing floor, and uses 1,440 minute bars per day for risk sizing. Policy holding horizons also retain physical duration.
- New forecasts refit weekly, use trailing 52-week training data, scale using training data only, and purge labels using their actual completion timestamp. The original strategy retains its trailing 12-week regression window. Features/labels spanning missing common quotes are invalidated for the new models. No full-sample fitting.
- Same-quote execution observes quote t, trades at its bid/ask and earns only t→t+1. A one-quote delay trades at t+1 and earns t+1→t+2. Delays count common valid quotes; across gaps they can exceed their nominal minute count. Returns across gaps remain in held-position PnL.
- Costs include every basket leg, each adjustment and final liquidation. Daily Sharpe uses the New York 17:00 trading-date convention and √252. Reported PnL uses fixed log-return notional accounting, not a cash settlement ledger. No fees, market impact or actual financing are included in base results.
- New policies request liquidation from 16:30 to 18:00 New York. The original strategy may hold through rollover; its results need financing data. A separate one-basis-point-per-rollover-notional stress is reported.

## Declared search and selection

Seventeen configurations: reversal lookbacks 5/15/30/120/480 minutes; equilibrium half-lives 15/60/240/960 minutes; and forecast no-trade bands at 30/60/240/720-minute label horizons with spread hurdles 1 and 2. Reversal/equilibrium entry z-score is fixed at 2. The three pre-existing M15 shortlist policies remain benchmarks.

2016 is warm-up. Rank by the minimum net Sharpe across 2017–2022, 2021–2022 and 2023, requiring at least 40 validation trade days. The last two development periods overlap by design to check recent behaviour. Freeze the top three, then score 2024–2025. Those final years were already inspected in previous research, so this is **not a fresh untouched holdout**. Earlier searches plus these 17 trials create additional selection uncertainty.

Frozen shortlist: band_240m_cost2, band_60m_cost1, equilibrium_60m. Among candidates with at least 40 validation trade days, positive net Sharpe in all three selection periods: none. The ensemble averages their basket targets and nets trades before costs.

| policy | development | recent_development | validation | score | validation_trade_days |
| --- | --- | --- | --- | --- | --- |
| band_60m_cost2 | 0.180 | 0.343 | 1.605 | 0.180 | 20 |
| equilibrium_240m | 0.372 | 0.170 | 1.099 | 0.170 | 21 |
| band_30m_cost1 | 0.150 | 0.266 | 0.980 | 0.150 | 25 |
| band_30m_cost2 | 0.281 | 0.093 | 1.366 | 0.093 | 2 |
| band_240m_cost2 | -0.093 | 0.583 | 1.299 | -0.093 | 63 |
| band_60m_cost1 | -0.095 | -0.097 | 1.402 | -0.097 | 62 |
| equilibrium_60m | -0.175 | 0.254 | 4.052 | -0.175 | 85 |
| band_720m_cost2 | 0.187 | 0.574 | -0.179 | -0.179 | 62 |
| band_240m_cost1 | -0.269 | -0.116 | 1.245 | -0.269 | 73 |
| equilibrium_960m | 0.195 | -0.305 | 0.984 | -0.305 | 1 |
| reversal_480m | 0.111 | -0.650 | 0.335 | -0.650 | 9 |
| reversal_120m | -0.951 | -0.037 | 1.359 | -0.951 | 72 |
| band_720m_cost1 | -0.154 | -0.134 | -1.250 | -1.250 | 73 |
| equilibrium_15m | -2.087 | -1.945 | -0.112 | -2.087 | 161 |
| reversal_30m | -4.465 | -5.193 | -3.544 | -5.193 | 177 |
| reversal_5m | -7.583 | -6.305 | -5.237 | -7.583 | 143 |
| reversal_15m | -8.202 | -8.982 | -6.863 | -8.982 | 224 |

## Evaluation: 2024–2025 net Sharpe

Columns are execution delays in common valid one-minute quotes. All models below generate targets every minute.

| policy | 0 | 1 | 2 | 15 |
| --- | --- | --- | --- | --- |
| band_240m_cost2 | -1.082 | -1.635 | -1.569 | -2.290 |
| band_60m_cost1 | -0.972 | -1.868 | -2.091 | -3.171 |
| band_720m_cost2 | -0.656 | -1.207 | -1.134 | -2.056 |
| equilibrium_240m | 0.221 | -0.234 | -0.540 | -1.022 |
| equilibrium_60m | 0.488 | -0.172 | -0.701 | -2.805 |
| frozen_ensemble | -0.354 | -1.209 | -1.601 | -3.333 |
| legacy_corrected | -1.542 | -2.456 | -2.567 | -3.740 |

| policy | development | validation | 2024 | 2025 | evaluation |
| --- | --- | --- | --- | --- | --- |
| band_240m_cost2 | -0.093 | 1.299 | -0.852 | -1.577 | -1.082 |
| band_60m_cost1 | -0.095 | 1.402 | -0.741 | -1.508 | -0.972 |
| band_720m_cost2 | 0.187 | -0.179 | -1.057 | -0.302 | -0.656 |
| equilibrium_240m | 0.372 | 1.099 | 0.866 | -0.110 | 0.221 |
| equilibrium_60m | -0.175 | 4.052 | 2.200 | -0.652 | 0.488 |
| frozen_ensemble | -0.151 | 3.533 | 0.358 | -1.181 | -0.354 |
| legacy_corrected | -0.451 | -6.451 | -1.269 | -1.804 | -1.542 |

The matched 15-minute controls below use quotes sampled at quarter-hours from the **same new minute cache**, fit/compute signals on that 15-minute grid, and execute using the full minute quote stream. This keeps source data fixed when comparing signal and execution frequency.

| policy | 0 | 1 | 15 |
| --- | --- | --- | --- |
| band_240m_cost2 | -0.866 | -1.253 | -1.442 |
| band_720m_cost2 | -0.949 | -1.488 | -2.161 |
| equilibrium_240m | -1.061 | -1.288 | -1.786 |
| legacy_corrected | -0.670 | -0.881 | -2.274 |

## Sensitivity and uncertainty

`stress_evaluation.csv` includes 1.5×/2× spreads, an extra 0.25 bp per traded leg, financing sensitivity, changing targets only every 15 minutes, and allowing fills only if every leg's recorded bid/ask changed within the preceding two minutes. The last condition is a **causal freshness proxy**, not actual tick age. It gates execution after the delay and preserves held exposure while a fill is blocked. Final liquidation is unconditional.

Forecast policies in the new shortlist were additionally refitted with one-minute-delayed labels; see `refitted_delay1.csv`. Delaying an immediate-return-trained target and retraining for the delayed horizon are reported separately.

| policy | fill_delay_minutes | scenario | sharpe |
| --- | --- | --- | --- |
| legacy_corrected | 0 | base | -1.542 |
| legacy_corrected | 0 | spread_x1.5 | -2.889 |
| legacy_corrected | 0 | spread_x2 | -3.712 |
| legacy_corrected | 0 | extra_0.25bp_per_leg | -3.383 |
| legacy_corrected | 0 | 1bp_per_roll_exposure | -1.881 |
| legacy_corrected | 1 | base | -2.456 |
| legacy_corrected | 1 | spread_x1.5 | -3.810 |
| legacy_corrected | 1 | spread_x2 | -4.565 |
| legacy_corrected | 1 | extra_0.25bp_per_leg | -4.466 |
| legacy_corrected | 1 | 1bp_per_roll_exposure | -2.838 |
| legacy_corrected | 0 | fresh_at_fill_2min | -1.484 |
| legacy_corrected | 1 | fresh_at_fill_2min | -2.484 |
| legacy_corrected | 0 | decisions_every_15min | -2.332 |
| legacy_corrected | 1 | decisions_every_15min | -2.508 |
| band_240m_cost2 | 0 | base | -1.082 |
| band_240m_cost2 | 0 | spread_x1.5 | -1.480 |
| band_240m_cost2 | 0 | spread_x2 | -1.838 |
| band_240m_cost2 | 0 | extra_0.25bp_per_leg | -1.551 |
| band_240m_cost2 | 0 | 1bp_per_roll_exposure | -1.082 |
| band_240m_cost2 | 1 | base | -1.635 |
| band_240m_cost2 | 1 | spread_x1.5 | -2.068 |
| band_240m_cost2 | 1 | spread_x2 | -2.435 |
| band_240m_cost2 | 1 | extra_0.25bp_per_leg | -2.091 |
| band_240m_cost2 | 1 | 1bp_per_roll_exposure | -1.635 |
| band_240m_cost2 | 0 | fresh_at_fill_2min | -1.085 |
| band_240m_cost2 | 1 | fresh_at_fill_2min | -1.638 |
| band_240m_cost2 | 0 | decisions_every_15min | -1.555 |
| band_240m_cost2 | 1 | decisions_every_15min | -1.901 |
| equilibrium_60m | 0 | base | 0.488 |
| equilibrium_60m | 0 | spread_x1.5 | -1.066 |
| equilibrium_60m | 0 | spread_x2 | -2.575 |
| equilibrium_60m | 0 | extra_0.25bp_per_leg | -1.131 |
| equilibrium_60m | 0 | 1bp_per_roll_exposure | 0.479 |
| equilibrium_60m | 1 | base | -0.172 |
| equilibrium_60m | 1 | spread_x1.5 | -1.753 |
| equilibrium_60m | 1 | spread_x2 | -3.211 |
| equilibrium_60m | 1 | extra_0.25bp_per_leg | -1.820 |
| equilibrium_60m | 1 | 1bp_per_roll_exposure | -0.183 |
| equilibrium_60m | 0 | fresh_at_fill_2min | 0.488 |
| equilibrium_60m | 1 | fresh_at_fill_2min | -0.172 |
| equilibrium_60m | 0 | decisions_every_15min | -1.030 |
| equilibrium_60m | 1 | decisions_every_15min | -1.332 |
| frozen_ensemble | 0 | base | -0.354 |
| frozen_ensemble | 0 | spread_x1.5 | -1.714 |
| frozen_ensemble | 0 | spread_x2 | -2.910 |
| frozen_ensemble | 0 | extra_0.25bp_per_leg | -1.866 |
| frozen_ensemble | 0 | 1bp_per_roll_exposure | -0.360 |
| frozen_ensemble | 1 | base | -1.209 |
| frozen_ensemble | 1 | spread_x1.5 | -2.571 |
| frozen_ensemble | 1 | spread_x2 | -3.684 |
| frozen_ensemble | 1 | extra_0.25bp_per_leg | -2.684 |
| frozen_ensemble | 1 | 1bp_per_roll_exposure | -1.217 |
| frozen_ensemble | 0 | fresh_at_fill_2min | -0.357 |
| frozen_ensemble | 1 | fresh_at_fill_2min | -1.211 |
| frozen_ensemble | 0 | decisions_every_15min | -1.917 |
| frozen_ensemble | 1 | decisions_every_15min | -2.266 |

Twenty-trading-day circular block bootstrap intervals use 2,000 draws. They measure sampling uncertainty conditional on the selected rules and data; they do not correct for the preceding strategy search, stale quotes or missing costs. `bootstrap.csv` contains all intervals. Variance ratios are descriptive, not a proof of stationarity or an executable edge. Regression diagnostics use 30-minute-spaced anchors, minute-resolved labels and HAC errors.

## Reproduce

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_minute.py --stage all
poetry run pytest -q tests/test_sgd_neer_backtest.py
```

The runner resumes from local `tmp/sgd_neer_m1_research` artifacts. After changing the model or source data, use a clean research cache/results directory; cached signals are not automatically invalidated. The original exploratory notebook is preserved. Open [sgd_neer_minute.ipynb](sgd_neer_minute.ipynb) for executed tables and plots. Full daily results, calibration fits, source audits and protocols are in [sgd_neer_results/minute](sgd_neer_results/minute).
