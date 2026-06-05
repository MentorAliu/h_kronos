from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.kronos_runner import resolve_manifest_path


METRIC_COLUMNS = [
    "timeframe",
    "windows",
    "lookback",
    "mae",
    "rmse",
    "directional_accuracy",
]


class EvaluationError(ValueError):
    """Raised when a clean candle dataset cannot be evaluated safely."""


@dataclass(frozen=True)
class EvaluationWindow:
    timeframe: str
    input_window: pd.DataFrame
    input_start_timestamp: str
    input_end_timestamp: str
    target_timestamp: str
    current_close: float
    target_close: float


@dataclass(frozen=True)
class BaselineEvaluationRun:
    manifest_path: Path
    output_path: Path
    rows: int


def build_evaluation_windows(
    clean: pd.DataFrame,
    *,
    timeframe: str,
    lookback: int,
) -> list[EvaluationWindow]:
    if lookback <= 0:
        raise EvaluationError("lookback must be positive")

    clean = _normalize_clean(clean, timeframe=timeframe)
    minimum_rows = lookback + 1
    if len(clean) < minimum_rows:
        raise EvaluationError(
            f"{timeframe} baseline evaluation requires at least {minimum_rows} clean rows; found {len(clean)}"
        )

    windows: list[EvaluationWindow] = []
    for target_index in range(lookback, len(clean)):
        input_window = clean.iloc[target_index - lookback : target_index].reset_index(drop=True)
        target = clean.iloc[target_index]
        input_end = pd.Timestamp(input_window["timestamp"].iloc[-1])
        target_timestamp = pd.Timestamp(target["timestamp"])
        if input_end >= target_timestamp:
            raise EvaluationError("Evaluation input window includes or overlaps its target")

        windows.append(
            EvaluationWindow(
                timeframe=timeframe,
                input_window=input_window,
                input_start_timestamp=_format_timestamp(input_window["timestamp"].iloc[0]),
                input_end_timestamp=_format_timestamp(input_end),
                target_timestamp=_format_timestamp(target_timestamp),
                current_close=float(input_window["close"].iloc[-1]),
                target_close=float(target["close"]),
            )
        )

    return windows


def naive_close_persistence_metrics(
    windows: list[EvaluationWindow],
    *,
    timeframe: str,
    lookback: int,
) -> dict[str, Any]:
    if not windows:
        raise EvaluationError(f"{timeframe} has no evaluation windows")

    current = np.array([window.current_close for window in windows], dtype=float)
    target = np.array([window.target_close for window in windows], dtype=float)
    errors = current - target
    baseline_direction = np.zeros_like(target)
    actual_direction = np.sign(target - current)

    return {
        "timeframe": timeframe,
        "windows": len(windows),
        "lookback": lookback,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "directional_accuracy": float(np.mean(baseline_direction == actual_direction)),
    }


def evaluate_baseline_manifest(
    *,
    manifest: str | Path,
    manifest_dir: Path,
    output_dir: Path,
    lookback: int,
) -> BaselineEvaluationRun:
    manifest_path = resolve_manifest_path(manifest, manifest_dir=manifest_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for dataset in payload.get("datasets", []):
        timeframe = dataset["timeframe"]
        clean_path = Path(dataset["clean_path"])
        if not clean_path.exists():
            raise EvaluationError(f"Clean candle file does not exist: {clean_path}")
        clean = pd.read_csv(clean_path)
        windows = build_evaluation_windows(clean, timeframe=timeframe, lookback=lookback)
        rows.append(
            naive_close_persistence_metrics(
                windows,
                timeframe=timeframe,
                lookback=lookback,
            )
        )

    if not rows:
        raise EvaluationError("Manifest contains no datasets to evaluate")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['run_id']}_naive_baseline_metrics.csv"
    pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(output_path, index=False)
    return BaselineEvaluationRun(
        manifest_path=manifest_path,
        output_path=output_path,
        rows=len(rows),
    )


def _normalize_clean(clean: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    if list(clean.columns) != CLEAN_COLUMNS:
        raise EvaluationError(
            f"{timeframe} clean candle schema mismatch; expected: " + ", ".join(CLEAN_COLUMNS)
        )

    normalized = clean.copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"],
        utc=True,
        errors="coerce",
    )
    if normalized["timestamp"].isna().any():
        raise EvaluationError(f"{timeframe} clean file contains invalid timestamps")

    for column in CLEAN_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[CLEAN_COLUMNS[1:]].isna().any().any():
        raise EvaluationError(f"{timeframe} clean file contains invalid numeric values")

    if not normalized["timestamp"].is_monotonic_increasing:
        raise EvaluationError(f"{timeframe} timestamps must be sorted ascending")

    return normalized


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
