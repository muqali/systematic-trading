from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
import json

import pandas as pd

from analytics.performance_report import PerformanceReport
from strategy.factor_mean_reversion_strategy import FactorMeanReversionStrategy
from trading.aggressive_trader import AggressiveTrader


@dataclass
class HyperParamOptimizationResult:
    best_params_by_pair: dict[str, dict]
    best_sharpe_by_pair: pd.DataFrame
    total_trials: int
    elapsed_seconds: float
    log_path: Path


def _flatten_report(report_df: pd.DataFrame) -> dict[str, float]:
    flat: dict[str, float] = {}
    for pair, row in report_df.iterrows():
        for metric, value in row.items():
            flat[f"{pair}__{metric}"] = value
    return flat


def _to_loggable_value(value):
    if isinstance(value, pd.Timedelta):
        return str(value)
    return value


def _append_rows_csv(rows: list[dict], log_path: Path) -> None:
    if not rows:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    batch_df = pd.DataFrame(rows)
    write_header = not log_path.exists()
    batch_df.to_csv(log_path, mode="a", header=write_header, index=False)


def _params_signature(params: dict) -> str:
    return json.dumps({k: _to_loggable_value(v) for k, v in params.items()}, sort_keys=True)


def _load_resume_state(
    instruments: list[str], log_path: Path
) -> tuple[set[str], dict[str, float], dict[str, dict]]:
    existing_signatures: set[str] = set()
    best_sharpe: dict[str, float] = {pair: float("-inf") for pair in instruments}
    best_params: dict[str, dict] = {pair: {} for pair in instruments}

    if not log_path.exists():
        return existing_signatures, best_sharpe, best_params

    existing = pd.read_csv(log_path)
    if "params_json" not in existing.columns:
        return existing_signatures, best_sharpe, best_params

    for _, row in existing.iterrows():
        params_json = row["params_json"]
        if pd.isna(params_json):
            continue

        params_json = str(params_json)
        existing_signatures.add(params_json)

        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            continue

        for pair in instruments:
            sharpe_col = f"{pair}__Sharpe_Net"
            if sharpe_col not in existing.columns:
                continue
            sharpe = row[sharpe_col]
            if pd.notna(sharpe) and sharpe > best_sharpe[pair]:
                best_sharpe[pair] = float(sharpe)
                best_params[pair] = params

    return existing_signatures, best_sharpe, best_params


def optimize_factor_mean_reversion_hyperparams(
    instruments: list[str],
    close_price_dict: dict[str, pd.DataFrame],
    param_grid: dict[str, list],
    log_path: str | Path = "src/systematic_trading/hyperparam_trials.csv",
    checkpoint_every: int = 10,
    resume: bool = True,
) -> HyperParamOptimizationResult:
    """
    Grid-search all hyperparameter combinations and checkpoint trial results.
    Returns the best params for each pair using highest Sharpe_Net.
    """

    keys = list(param_grid.keys())
    combinations = list(product(*(param_grid[k] for k in keys)))
    log_path = Path(log_path)

    start_total = perf_counter()
    rows_buffer: list[dict] = []

    if resume:
        existing_signatures, best_sharpe, best_params = _load_resume_state(
            instruments, log_path
        )
    else:
        existing_signatures = set()
        best_sharpe = {pair: float("-inf") for pair in instruments}
        best_params = {pair: {} for pair in instruments}

    try:
        for trial_id, vals in enumerate(combinations, start=1):
            trial_start = perf_counter()
            params = dict(zip(keys, vals))
            params_json = _params_signature(params)
            if params_json in existing_signatures:
                continue

            strategy = FactorMeanReversionStrategy(
                instruments=tuple(instruments),
                close_price_dict=close_price_dict,
                hyper_param_dict=params,
            )
            signals = strategy.generate_signals()

            trader = AggressiveTrader(
                instruments=instruments,
                close_price_dict=close_price_dict,
                signals=signals,
            )
            report = PerformanceReport(instruments=instruments, trader=trader).generate_performance_report()

            trial_elapsed = perf_counter() - trial_start
            elapsed_total = perf_counter() - start_total

            row = {
                "trial_id": trial_id,
                "trial_elapsed_seconds": trial_elapsed,
                "elapsed_total_seconds": elapsed_total,
                "params_json": params_json,
            }
            row.update({f"param__{k}": _to_loggable_value(v) for k, v in params.items()})
            row.update(_flatten_report(report))
            rows_buffer.append(row)
            existing_signatures.add(params_json)

            for pair in instruments:
                sharpe = report.loc[pair, "Sharpe_Net"]
                if pd.notna(sharpe) and sharpe > best_sharpe[pair]:
                    best_sharpe[pair] = float(sharpe)
                    best_params[pair] = dict(params)

            if checkpoint_every > 0 and (trial_id % checkpoint_every == 0):
                _append_rows_csv(rows_buffer, log_path)
                rows_buffer.clear()
    finally:
        _append_rows_csv(rows_buffer, log_path)
        rows_buffer.clear()

    elapsed_seconds = perf_counter() - start_total
    best_sharpe_df = pd.DataFrame(
        {
            "best_sharpe_net": pd.Series(best_sharpe),
            "best_params": pd.Series(best_params),
        }
    ).sort_index()

    return HyperParamOptimizationResult(
        best_params_by_pair=best_params,
        best_sharpe_by_pair=best_sharpe_df,
        total_trials=len(combinations),
        elapsed_seconds=elapsed_seconds,
        log_path=log_path,
    )
