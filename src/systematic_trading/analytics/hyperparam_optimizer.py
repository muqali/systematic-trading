from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
import json
import math

import pandas as pd

from analytics.performance_report import PerformanceReport
from strategy.factor_mean_reversion_strategy import FactorMeanReversionStrategy
from trading.aggressive_trader import AggressiveTrader


@dataclass
class HyperParamOptimizationResult:
    best_params_by_pair: dict[str, dict]
    best_metric_by_pair: pd.DataFrame
    best_params_portfolio: dict
    best_metric_portfolio: pd.Series
    metric_name: str
    total_trials: int
    completed_trials: int
    elapsed_seconds: float
    log_path: Path

    @property
    def best_sharpe_by_pair(self) -> pd.DataFrame:
        return self.best_metric_by_pair


def _flatten_report(report_df: pd.DataFrame) -> dict[str, float]:
    flat: dict[str, float] = {}
    for pair, row in report_df.iterrows():
        for metric, value in row.items():
            flat[f"{pair}__{metric}"] = value
    return flat


def _serialize_param_value(value):
    if isinstance(value, pd.Timedelta):
        return {"__type__": "Timedelta", "value": value.isoformat()}
    if isinstance(value, pd.Timestamp):
        return {"__type__": "Timestamp", "value": value.isoformat()}
    if isinstance(value, Path):
        return {"__type__": "Path", "value": str(value)}
    if isinstance(value, tuple):
        return {
            "__type__": "Tuple",
            "value": [_serialize_param_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_serialize_param_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_param_value(item) for key, item in value.items()
        }
    return value


def _deserialize_param_value(value):
    if isinstance(value, list):
        return [_deserialize_param_value(item) for item in value]
    if not isinstance(value, dict) or "__type__" not in value:
        return value

    value_type = value["__type__"]
    if value_type == "Timedelta":
        return pd.Timedelta(value["value"])
    if value_type == "Timestamp":
        return pd.Timestamp(value["value"])
    if value_type == "Path":
        return Path(value["value"])
    if value_type == "Tuple":
        return tuple(_deserialize_param_value(item) for item in value["value"])
    return value


def _to_loggable_value(value):
    if isinstance(value, (pd.Timedelta, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, tuple):
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
    serialized = {k: _serialize_param_value(v) for k, v in params.items()}
    return json.dumps(serialized, sort_keys=True)


def _parse_params_json(params_json: str) -> dict:
    raw_params = json.loads(params_json)
    return {k: _deserialize_param_value(v) for k, v in raw_params.items()}


def _resolve_metric_name(
    available_metrics: list[str] | pd.Index, requested_metric: str
) -> str:
    if requested_metric in available_metrics:
        return requested_metric

    requested_key = requested_metric.casefold()
    matches = [
        metric for metric in available_metrics if str(metric).casefold() == requested_key
    ]
    if len(matches) == 1:
        return str(matches[0])

    raise ValueError(
        f"Metric '{requested_metric}' was not found. "
        f"Available metrics: {list(available_metrics)}"
    )


def _load_resume_state(
    instruments: list[str],
    log_path: Path,
    requested_metric: str,
    portfolio_label: str,
) -> tuple[set[str], str | None, dict[str, float], dict[str, dict], float, dict]:
    existing_signatures: set[str] = set()
    resolved_metric: str | None = None
    best_metric: dict[str, float] = {pair: float("-inf") for pair in instruments}
    best_params: dict[str, dict] = {pair: {} for pair in instruments}
    best_portfolio_metric = float("-inf")
    best_portfolio_params: dict = {}

    if not log_path.exists():
        return (
            existing_signatures,
            resolved_metric,
            best_metric,
            best_params,
            best_portfolio_metric,
            best_portfolio_params,
        )

    existing = pd.read_csv(log_path)
    if "params_json" not in existing.columns:
        return (
            existing_signatures,
            resolved_metric,
            best_metric,
            best_params,
            best_portfolio_metric,
            best_portfolio_params,
        )

    metric_suffixes = []
    for instrument in instruments:
        prefix = f"{instrument}__"
        metric_suffixes.extend(
            column[len(prefix):]
            for column in existing.columns
            if column.startswith(prefix)
        )
    portfolio_prefix = f"{portfolio_label}__"
    metric_suffixes.extend(
        column[len(portfolio_prefix):]
        for column in existing.columns
        if column.startswith(portfolio_prefix)
    )
    if metric_suffixes:
        resolved_metric = _resolve_metric_name(metric_suffixes, requested_metric)

    for _, row in existing.iterrows():
        params_json = row["params_json"]
        if pd.isna(params_json):
            continue

        params_json = str(params_json)
        existing_signatures.add(params_json)

        try:
            params = _parse_params_json(params_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        if resolved_metric is None:
            continue

        for pair in instruments:
            metric_col = f"{pair}__{resolved_metric}"
            if metric_col not in existing.columns:
                continue
            score = row[metric_col]
            if pd.notna(score) and score > best_metric[pair]:
                best_metric[pair] = float(score)
                best_params[pair] = params

        portfolio_metric_col = f"{portfolio_label}__{resolved_metric}"
        if portfolio_metric_col in existing.columns:
            score = row[portfolio_metric_col]
            if pd.notna(score) and score > best_portfolio_metric:
                best_portfolio_metric = float(score)
                best_portfolio_params = params

    return (
        existing_signatures,
        resolved_metric,
        best_metric,
        best_params,
        best_portfolio_metric,
        best_portfolio_params,
    )


def optimize_strategy_hyperparams(
    strategy_cls,
    strategy_kwargs: dict,
    instruments: list[str],
    close_price_dict: dict[str, pd.DataFrame],
    param_grid: dict[str, list],
    trader_cls=AggressiveTrader,
    trader_kwargs: dict | None = None,
    metric: str = "Sharpe_Net",
    portfolio_label: str = PerformanceReport.PORTFOLIO_LABEL,
    log_path: str | Path = "hyperparam_trials.csv",
    checkpoint_every: int = 10,
    resume: bool = True,
) -> HyperParamOptimizationResult:
    """
    Grid-search all hyperparameter combinations for any strategy that accepts
    ``hyper_param_dict`` in its constructor.

    The optimizer records every completed trial to CSV, can resume from that log,
    and returns the best parameter set for each instrument and for the full
    portfolio based on the selected PerformanceReport metric.
    """

    strategy_kwargs = dict(strategy_kwargs)
    if "hyper_param_dict" in strategy_kwargs:
        raise ValueError(
            "Pass static constructor arguments in strategy_kwargs only. "
            "The optimizer manages hyper_param_dict from param_grid."
        )

    trader_kwargs = dict(trader_kwargs or {})
    if "signals" in trader_kwargs:
        raise ValueError(
            "Do not include 'signals' in trader_kwargs. "
            "The optimizer injects strategy-generated signals."
        )

    keys = list(param_grid.keys())
    total_trials = math.prod(len(param_grid[key]) for key in keys) if keys else 1
    combinations = product(*(param_grid[key] for key in keys)) if keys else [()]
    log_path = Path(log_path)

    start_total = perf_counter()
    rows_buffer: list[dict] = []

    if resume:
        (
            existing_signatures,
            resolved_metric,
            best_metric,
            best_params,
            best_portfolio_metric,
            best_portfolio_params,
        ) = _load_resume_state(instruments, log_path, metric, portfolio_label)
    else:
        existing_signatures = set()
        resolved_metric = None
        best_metric = {pair: float("-inf") for pair in instruments}
        best_params = {pair: {} for pair in instruments}
        best_portfolio_metric = float("-inf")
        best_portfolio_params = {}

    try:
        for trial_id, vals in enumerate(combinations, start=1):
            trial_start = perf_counter()
            params = dict(zip(keys, vals))
            params_json = _params_signature(params)
            if params_json in existing_signatures:
                continue

            strategy = strategy_cls(
                **strategy_kwargs,
                hyper_param_dict=params,
            )
            signals = strategy.generate_signals()

            effective_trader_kwargs = dict(trader_kwargs)
            effective_trader_kwargs["instruments"] = list(
                effective_trader_kwargs.get("instruments", instruments)
            )
            effective_trader_kwargs["close_price_dict"] = effective_trader_kwargs.get(
                "close_price_dict", close_price_dict
            )
            effective_trader_kwargs["signals"] = signals

            trader = trader_cls(**effective_trader_kwargs)
            report = PerformanceReport(
                instruments=effective_trader_kwargs["instruments"],
                trader=trader,
            ).generate_performance_report(
                include_portfolio=True,
                portfolio_label=portfolio_label,
            )

            if resolved_metric is None:
                resolved_metric = _resolve_metric_name(report.columns, metric)

            trial_elapsed = perf_counter() - trial_start
            elapsed_total = perf_counter() - start_total

            row = {
                "trial_id": trial_id,
                "trial_elapsed_seconds": trial_elapsed,
                "elapsed_total_seconds": elapsed_total,
                "metric_name": resolved_metric,
                "params_json": params_json,
            }
            row.update(
                {f"param__{key}": _to_loggable_value(value) for key, value in params.items()}
            )
            row.update(_flatten_report(report))
            rows_buffer.append(row)
            existing_signatures.add(params_json)

            for pair in instruments:
                score = report.loc[pair, resolved_metric]
                if pd.notna(score) and score > best_metric[pair]:
                    best_metric[pair] = float(score)
                    best_params[pair] = dict(params)

            portfolio_score = report.loc[portfolio_label, resolved_metric]
            if pd.notna(portfolio_score) and portfolio_score > best_portfolio_metric:
                best_portfolio_metric = float(portfolio_score)
                best_portfolio_params = dict(params)

            if checkpoint_every > 0 and (trial_id % checkpoint_every == 0):
                _append_rows_csv(rows_buffer, log_path)
                rows_buffer.clear()
    finally:
        _append_rows_csv(rows_buffer, log_path)
        rows_buffer.clear()

    if resolved_metric is None:
        resolved_metric = metric

    elapsed_seconds = perf_counter() - start_total
    best_metric_by_pair = pd.DataFrame(
        {
            "best_metric": pd.Series(best_metric),
            "best_params": pd.Series(best_params),
        }
    ).sort_index()
    best_metric_portfolio = pd.Series(
        {
            "best_metric": best_portfolio_metric,
            "best_params": best_portfolio_params,
        },
        name=portfolio_label,
    )

    return HyperParamOptimizationResult(
        best_params_by_pair=best_params,
        best_metric_by_pair=best_metric_by_pair,
        best_params_portfolio=best_portfolio_params,
        best_metric_portfolio=best_metric_portfolio,
        metric_name=resolved_metric,
        total_trials=total_trials,
        completed_trials=len(existing_signatures),
        elapsed_seconds=elapsed_seconds,
        log_path=log_path,
    )


def optimize_factor_mean_reversion_hyperparams(
    instruments: list[str],
    close_price_dict: dict[str, pd.DataFrame],
    param_grid: dict[str, list],
    log_path: str | Path = "src/systematic_trading/hyperparam_trials.csv",
    checkpoint_every: int = 10,
    resume: bool = True,
    metric: str = "Sharpe_Net",
) -> HyperParamOptimizationResult:
    return optimize_strategy_hyperparams(
        strategy_cls=FactorMeanReversionStrategy,
        strategy_kwargs={
            "instruments": tuple(instruments),
            "close_price_dict": close_price_dict,
        },
        instruments=instruments,
        close_price_dict=close_price_dict,
        param_grid=param_grid,
        trader_cls=AggressiveTrader,
        metric=metric,
        log_path=log_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )
