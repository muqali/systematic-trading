"""Verify selected holding-study configurations against the public strategy."""
from pathlib import Path
import gc
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.sgd_neer_codex.run_sgd_neer_minute import CACHE
from strategy.sgd_neer_equilibrium_strategy import SGDNEEREquilibriumStrategy

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"sgd_neer_results/holding_time"


def main():
    frozen=json.loads((RESULTS/"frozen_selection.json").read_text())
    meta=pd.read_csv(RESULTS/"policies.csv").set_index("policy")
    choices=list(dict.fromkeys(name for key in ["primary","aligned_risk_matched","holding_only"]
                              for name in frozen["selected"][key][:1]))
    data=pd.read_pickle(CACHE/"data.pkl")
    prices={p:pd.DataFrame({"mid":data.mids[p],
                            "bid":data.mids[p]*(1-data.spreads[p]/2),
                            "ask":data.mids[p]*(1+data.spreads[p]/2)}) for p in data.mids}
    audit=[]
    for name in choices:
        m=meta.loc[name]
        params=dict(freq="1min",equilibrium_half_life=pd.Timedelta(hours=int(m.horizon_hours)),
                    zscore_entry_threshold=float(m.entry_z),
                    max_holding_period=("unlimited" if m.holding_hours=="unlimited"
                                        else pd.Timedelta(hours=float(m.holding_hours))),
                    constituent_selection=m.rule,sizing_mode=m.sizing)
        strategy=SGDNEEREquilibriumStrategy(list(data.mids.columns),prices,params)
        print("Verifying full-history constructor:",name,flush=True)
        strategy.generate_signals()
        expected=pd.read_pickle(RESULTS/"targets"/f"{name}.pkl")
        pd.testing.assert_frame_equal(strategy.pair_targets,expected,atol=1e-9,rtol=1e-9)
        error=float(np.abs(strategy.pair_targets.to_numpy()-expected.to_numpy()).max())
        audit.append(dict(policy=name,quote_rows=len(data.mids),entry_count=len(strategy.entry_details),
                          max_pair_target_error=error))
        print(audit[-1],flush=True)
        del strategy
        gc.collect()
    (RESULTS/"strategy_verification.json").write_text(json.dumps(audit,indent=2)+"\n")


if __name__=="__main__":
    main()
