from itertools import product
import pandas as pd

def sweep_parameters(param_grid, objective_fn):

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    rows = []

    for combo in product(*values):

        params = dict(zip(keys, combo))

        metrics = objective_fn(**params)

        row = {**params, **metrics}

        rows.append(row)

    return pd.DataFrame(rows)