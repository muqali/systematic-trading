"""Build tables, charts and constructor configurations from the holding study."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"sgd_neer_results/holding_time"


def table(frame, digits=3):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "—"
        return str(value)
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(map(str,frame.columns)) + " |",
                      "| " + " | ".join(["---"]*len(frame.columns)) + " |",
                      *["| " + " | ".join(row) + " |" for row in rows]])


def main():
    frame = pd.read_csv(RESULTS/"comparison.csv")
    annual = pd.read_csv(RESULTS/"annual_pnl.csv")
    meta = pd.read_csv(RESULTS/"policies.csv").set_index("policy")
    board = pd.read_csv(RESULTS/"leaderboard_development.csv", index_col=0)
    held = pd.read_csv(RESULTS/"holding_statistics.csv")
    frozen = json.loads((RESULTS/"frozen_selection.json").read_text())
    baseline = frozen["baseline"]
    same = frame[frame.fill_delay_minutes.eq(0)]
    score = same.pivot(index="policy", columns="period", values="sharpe")
    final = same[same.period.eq("evaluation")].set_index("policy")
    delayed = frame[(frame.fill_delay_minutes == 1) & frame.period.eq("evaluation")].set_index("policy")
    final_hold = held[held.period.eq("evaluation")].set_index("policy")
    limits = ["48", "72", "120", "240", "480", "unlimited"]
    base_family = meta[(meta.horizon_hours == 24) & (meta.entry_z == 1.5)
                       & meta.rule.eq("aligned") & meta.sizing.eq("risk_matched")]
    fixed_names = [base_family[base_family.holding_hours.astype(str).eq(limit)].index[0] for limit in limits]
    fixed = score.loc[fixed_names,["development", "recent_development", "validation", "weak_2019_2021", "evaluation", "all_2017_2025"]].copy()
    fixed.insert(0,"max_hold_hours",limits)
    fixed["evaluation_delay1"] = delayed.loc[fixed_names,"sharpe"]
    fixed["evaluation_annual_cost_pct"] = final.loc[fixed_names,"annual_cost"]*100
    fixed["evaluation_drawdown_pct"] = final.loc[fixed_names,"max_drawdown"]*100
    fixed["evaluation_trades"] = final.loc[fixed_names,"completed_trades"]
    fixed["evaluation_mean_hold_h"] = final_hold.loc[fixed_names,"mean_hours"]
    fixed["evaluation_max_hold_h"] = final_hold.loc[fixed_names,"max_hours"]
    fixed.to_csv(RESULTS/"fixed_strategy_holding_comparison.csv")

    labelled = [("Original strategy",baseline)]
    for key,label in [("holding_only","Holding-only selection"),("primary","Joint-grid selection"),
                       ("aligned_risk_matched","Aligned/risk-matched selection"),("fixed_48h","Best with 48h cap")]:
        if frozen["selected"][key]:
            labelled.append((label,frozen["selected"][key][0]))
    hindsight = final.sharpe.idxmax()
    labelled.append(("Highest evaluation Sharpe (retrospective)",hindsight))
    rows, configs = [], {}
    for label,name in labelled:
        m = meta.loc[name]
        row = dict(choice=label, policy=name, half_life_h=m.horizon_hours, entry_z=m.entry_z,
                   max_hold_h=m.holding_hours, constituents=m.rule, sizing=m.sizing,
                   selection_score=board.loc[name,"score"],
                   development=score.loc[name,"development"], validation=score.loc[name,"validation"],
                   evaluation=score.loc[name,"evaluation"], evaluation_delay1=delayed.loc[name,"sharpe"],
                   weak_2019_2021=score.loc[name,"weak_2019_2021"], all_2017_2025=score.loc[name,"all_2017_2025"],
                   evaluation_vol_pct=final.loc[name,"annual_vol"]*100,
                   evaluation_drawdown_pct=final.loc[name,"max_drawdown"]*100,
                   evaluation_mean_hold_h=final_hold.loc[name,"mean_hours"],
                   evaluation_max_hold_h=final_hold.loc[name,"max_hours"])
        rows.append(row)
        configs[label] = dict(freq="1min", equilibrium_half_life=f"{m.horizon_hours:g}h",
                             zscore_entry_threshold=float(m.entry_z), zscore_exit_threshold=.25,
                             max_holding_period=("unlimited" if m.holding_hours == "unlimited" else f"{float(m.holding_hours):g}h"),
                             constituent_selection=m.rule, sizing_mode=m.sizing)
    choices = pd.DataFrame(rows)
    choices.to_csv(RESULTS/"selected_comparison.csv", index=False)
    (RESULTS/"selected_hyperparameters.json").write_text(json.dumps(configs,indent=2)+"\n")

    reoptimized = []
    for limit in limits:
        eligible = board[board.eligible & board.holding_hours.astype(str).eq(limit)]
        if eligible.empty:
            continue
        name = eligible.index[0]
        m = meta.loc[name]
        reoptimized.append(dict(max_hold_h=limit, policy=name, half_life_h=m.horizon_hours,
                                entry_z=m.entry_z, constituents=m.rule, sizing=m.sizing,
                                score=board.loc[name,"score"], evaluation=score.loc[name,"evaluation"],
                                evaluation_delay1=delayed.loc[name,"sharpe"]))
    reoptimized = pd.DataFrame(reoptimized)
    reoptimized.to_csv(RESULTS/"best_parameters_by_holding_limit.csv", index=False)

    controls = ["equilibrium_24h_z1.5_hold48h__aligned__risk_matched",
                "equilibrium_12h_z1.5_hold48h__aligned__risk_matched",
                "equilibrium_12h_z1.5_hold72h__aligned__risk_matched",
                "equilibrium_12h_z1.5_hold48h__aligned__retained",
                "equilibrium_12h_z1.5_hold72h__aligned__retained"]
    controlled = meta.loc[controls,["horizon_hours","entry_z","holding_hours","sizing"]].copy()
    for period in ["weak_2019_2021","evaluation","all_2017_2025"]:
        controlled[period] = score.loc[controls,period]
    controlled["evaluation_delay1"] = delayed.loc[controls,"sharpe"]
    controlled.to_csv(RESULTS/"controlled_parameter_comparison.csv")
    stress = pd.read_csv(RESULTS/"stress.csv")
    displayed_stress = stress[(stress.delay==0)&stress.policy.isin(choices.policy)].pivot(
        index="policy",columns="scenario",values="sharpe")

    fig,axes = plt.subplots(2,2,figsize=(13,9),layout="constrained")
    x=np.arange(len(limits))
    labels=[f"{s}h" if s!="unlimited" else "No cap" for s in limits]
    for period,color,label in [("development","#788696","2017–2022"),
                                ("weak_2019_2021","#be683e","2019–2021"),
                                ("evaluation","#176b8b","2024–2025")]:
        axes[0,0].plot(x,fixed[period],marker="o",label=label,color=color)
    axes[0,0].set(title="Change only the holding limit: 24h EWMA / 1.5z",ylabel="Net Sharpe",xticks=x,xticklabels=labels)
    axes[0,0].axhline(0,color="gray",linewidth=.7)
    axes[0,0].legend(frameon=False)
    axes[0,1].bar(x-.18,fixed.evaluation_annual_cost_pct,width=.36,color="#bb8062",label="Annual spread cost (%)")
    axes[0,1].bar(x+.18,-fixed.evaluation_drawdown_pct,width=.36,color="#899eae",label="Maximum drawdown (%)")
    axes[0,1].set(title="Costs and drawdowns: fixed strategy, 2024–2025",xticks=x,xticklabels=labels)
    axes[0,1].legend(frameon=False)
    colors=["#82929f","#176b8b","#42846a"]
    chart_names=list(dict.fromkeys([baseline,*frozen["selected"]["holding_only"][:1],*frozen["selected"]["primary"][:1]]))
    for name,color in zip(chart_names,colors):
        label=next(label for label,policy in labelled if policy==name)
        daily=pd.read_csv(RESULTS/"daily"/f"{name}_fill0.csv.gz",index_col=0,parse_dates=True)
        axes[1,0].plot(daily.index,daily.net.cumsum()*100,label=label,color=color)
        part=annual[(annual.policy==name)&(annual.delay==0)]
        axes[1,1].plot(part.year,part.sharpe,marker="o",label=label,color=color)
    axes[1,0].axvspan(pd.Timestamp("2019-01-01"),pd.Timestamp("2022-01-01"),color="#dba869",alpha=.15)
    axes[1,0].set(title="Daily mark-to-market, including held losses",ylabel="Cumulative PnL (percentage points)")
    axes[1,0].legend(frameon=False)
    axes[1,1].set(title="Annual net Sharpe of pre-2024 selections",xticks=list(range(2016,2026)))
    axes[1,1].axhline(0,color="gray",linewidth=.7)
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("SGD NEER: longer holding limits and parameter selection",fontsize=16)
    fig.supxlabel("Same-quote fills; recorded spreads; no financing. Market hours exclude weekend closure. 2024–2025 was previously inspected.",fontsize=9)
    fig.savefig(RESULTS/"holding_time_comparison.png",dpi=160)
    plt.close(fig)

    columns=["choice","half_life_h","entry_z","max_hold_h","constituents","sizing","selection_score","evaluation","evaluation_delay1"]
    uncertainty=pd.read_csv(RESULTS/"bootstrap.csv")
    report=f'''# SGD NEER: holding-time limits and winning parameters

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

{table(fixed[["max_hold_hours","development","validation","weak_2019_2021","evaluation","evaluation_delay1","all_2017_2025"]].reset_index(drop=True))}

Actual holdings and cost/risk outcomes in 2024–2025:

{table(fixed[["max_hold_hours","evaluation_trades","evaluation_mean_hold_h","evaluation_max_hold_h","evaluation_annual_cost_pct","evaluation_drawdown_pct"]].reset_index(drop=True))}

Holding statistics assign complete episodes to their entry year; daily PnL
and completed-trade counts follow calendar trading dates. A cap counts
scheduled market hours, with missing minutes included and weekend closure
excluded. A time exit without quotes fills at the next available quote.
Unlimited disables the time exit; the mean exit and sample-end liquidation
remain. Identical results at large caps indicate that those caps never bind
on this history, not independent evidence for a precisely optimal duration.

## Joint selection and retrospective comparison

{table(choices[columns])}

The retrospective row maximises 2024–2025 Sharpe after seeing that period;
it is shown explicitly apart from the choices selected on earlier data.
Equivalent parameter settings can tie when their time limits never bind.
Normalized subset sizing keeps aggregate SGD exposure but can increase
estimated risk; risk matching controls estimated entry risk, not realised risk.

Risk and weak-period performance of these choices:

{table(choices[["choice","weak_2019_2021","all_2017_2025","evaluation_vol_pct","evaluation_drawdown_pct","evaluation_mean_hold_h","evaluation_max_hold_h"]])}

Controlled comparisons separating the faster reference, holding cap and sizing:

{table(controlled.reset_index(drop=True))}

## Re-optimise other parameters at each holding cap

Each row below is chosen using only the pre-2024 ranking, then evaluated.

{table(reoptimized.drop(columns="policy"))}

## Uncertainty and execution costs

Conditional 95% Sharpe intervals and paired differences from the original
24h/1.5z/48h strategy use 2,000 shared 20-day block resamples. They do not
adjust for the expanded strategy search or the previously inspected dates.

{table(uncertainty[uncertainty.policy.isin(choices.policy)][["policy","low","high","delta_low","delta_high"]])}

`stress.csv` contains next-quote execution, 1.5×/2× spread costs and an extra
0.25 bp per quoted leg traded. 2023 common-quote coverage is incomplete, and
the source cache does not preserve tick-age information. These limitations
remain material for both selection and evaluation.

Same-quote evaluation Sharpe under cost stresses:

{table(displayed_stress.reset_index())}

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
'''
    (HERE/"sgd_neer_holding_report.md").write_text(report)
    print("FIXED STRATEGY\n",fixed.round(4).to_string())
    print("CHOICES\n",choices.round(4).to_string(index=False))
    print("BEST AT EACH CAP\n",reoptimized.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
