# SGD NEER research

Research modules, runners, companion notebooks, reports and saved results from
the SGD NEER analysis are collected here.

| Study | Notebook | Description |
| --- | --- | --- |
| Initial cost-aware backtests | [sgd_neer_cost_aware.ipynb](sgd_neer_cost_aware.ipynb) | [Research report](sgd_neer_research_report.md), [same-bar execution](sgd_neer_same_bar_report.md) |
| One-minute quotes | [sgd_neer_minute.ipynb](sgd_neer_minute.ipynb) | [Minute report](sgd_neer_minute_report.md) |
| Multi-day horizons | [sgd_neer_multiday.ipynb](sgd_neer_multiday.ipynb) | [Multi-day report](sgd_neer_multiday_report.md) |
| Entry times and continuous rolling signals | [sgd_neer_timing.ipynb](sgd_neer_timing.ipynb) | [Timing report](sgd_neer_timing_report.md) |
| Aligned constituents and other subsets | [sgd_neer_subset.ipynb](sgd_neer_subset.ipynb) | [Subset report](sgd_neer_subset_report.md) |
| Performance across calendar periods | [Diagnostic chart and tables](sgd_neer_results/regime_diagnostics/) | [2019–2021 underperformance](sgd_neer_regime_report.md) |
| Longer and unlimited holding | [Parameter grid and saved backtests](sgd_neer_results/holding_time/) | [Holding-time report](sgd_neer_holding_report.md) |
| 20% tighter bid/ask spreads | [Fixed-trade and regenerated backtests](sgd_neer_results/spread_sensitivity/) | [24h equilibrium spread sensitivity](sgd_neer_spread_sensitivity_report.md) |
| Entry time, weekday and date within month | [Trade calendar diagnostics](sgd_neer_results/calendar_diagnostics/) | [Calendar-pattern report](sgd_neer_calendar_report.md) |
| Entry hour in quote-data timezone | [US/Eastern hourly diagnostics](sgd_neer_results/hourly_eastern/) | [Hourly Eastern-time report](sgd_neer_hourly_eastern_report.md) |

[sgd_neer_results/](sgd_neer_results/) contains the saved tables, daily PnL,
trade records, protocols and manifests. The relocation preserves numerical
results and historical manifests; paths and hashes recorded in those manifests
describe the code at the time of each run, before this folder was created.

## Code and imports

The research implementations are `sgd_neer_backtest.py`, `sgd_neer_multiday.py`,
`sgd_neer_timing.py` and `sgd_neer_subset.py`. With `src/systematic_trading` on
the Python import path, use the new package prefix:

```python
from research.sgd_neer_codex.sgd_neer_backtest import daily_pnl
from research.sgd_neer_codex.sgd_neer_timing import RollingPolicy, rolling_signal
from research.sgd_neer_codex.sgd_neer_subset import PairAccountant
```

Run the `run_sgd_neer_*.py` scripts from the repository root. For example:

```sh
poetry run python src/systematic_trading/research/sgd_neer_codex/run_sgd_neer_subset.py
```

The reports give the commands and prerequisites for each study. Local working
caches and report-generation helpers remain in the repository's
`tmp/sgd_neer_*` directories. Importing this package also enables reading the
existing `BasketData` pickle caches, which reference the former module path.

## Original notebook and reusable strategy

The original [sgd_neer.ipynb](../sgd_neer.ipynb) and
[sgd_neer.py](../sgd_neer.py) remain in their original locations. The notebook's
24-hour equilibrium example uses
[SGDNEEREquilibriumStrategy](../../strategy/sgd_neer_equilibrium_strategy.py),
which remains under `strategy/`; its configurable constructor and behavior are
unchanged. Automated tests remain under the repository's `tests/` directory.
