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
FORECAST_REQUIRED_COLUMNS = [
    "run_id",
    "forecast_created_at",
    "exchange",
    "symbol",
    "timeframe",
    "model_name",
    "tokenizer_name",
    "device",
    "lookback",
    "pred_len",
    "input_start_timestamp",
    "input_end_timestamp",
    "forecast_timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
FORECAST_EVALUATION_COLUMNS = [
    "run_id",
    "forecast_created_at",
    "exchange",
    "symbol",
    "timeframe",
    "model_name",
    "lookback",
    "pred_len",
    "input_end_timestamp",
    "forecast_timestamp",
    "current_close",
    "target_close",
    "kronos_close",
    "kronos_close_error",
    "kronos_absolute_error",
    "kronos_squared_error",
    "actual_return",
    "forecasted_return",
    "kronos_direction_hit",
    "naive_close",
    "naive_close_error",
    "naive_absolute_error",
    "naive_squared_error",
    "naive_return",
    "naive_direction_hit",
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


@dataclass(frozen=True)
class ForecastEvaluationRun:
    forecast_path: Path
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


def evaluate_forecast_manifest(
    *,
    forecast: str | Path,
    manifest: str | Path,
    manifest_dir: Path,
    output_dir: Path,
) -> ForecastEvaluationRun:
    forecast_path = Path(forecast)
    if not forecast_path.exists():
        raise EvaluationError(f"Forecast file does not exist: {forecast_path}")

    manifest_path = resolve_manifest_path(manifest, manifest_dir=manifest_dir)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    forecast_rows = _normalize_forecast(pd.read_csv(forecast_path))
    clean_by_timeframe = _load_clean_by_timeframe(manifest_payload)

    rows = [
        _evaluate_forecast_row(
            forecast_row=forecast_row,
            clean_by_timeframe=clean_by_timeframe,
        )
        for _, forecast_row in forecast_rows.iterrows()
    ]
    if not rows:
        raise EvaluationError("Forecast file contains no rows to evaluate")

    run_ids = sorted({str(row["run_id"]) for row in rows})
    output_run_id = run_ids[0] if len(run_ids) == 1 else "mixed_forecast_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_run_id}_forecast_metrics.csv"
    pd.DataFrame(rows, columns=FORECAST_EVALUATION_COLUMNS).to_csv(output_path, index=False)
    return ForecastEvaluationRun(
        forecast_path=forecast_path,
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


def _normalize_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    if list(forecast.columns) != FORECAST_REQUIRED_COLUMNS:
        raise EvaluationError(
            "Forecast schema mismatch; expected: " + ", ".join(FORECAST_REQUIRED_COLUMNS)
        )

    normalized = forecast.copy()
    for column in (
        "forecast_created_at",
        "input_start_timestamp",
        "input_end_timestamp",
        "forecast_timestamp",
    ):
        normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
    timestamp_columns = [
        "forecast_created_at",
        "input_start_timestamp",
        "input_end_timestamp",
        "forecast_timestamp",
    ]
    if normalized[timestamp_columns].isna().any().any():
        raise EvaluationError("Forecast file contains invalid timestamps")

    for column in ("lookback", "pred_len", "open", "high", "low", "close", "volume", "amount"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    numeric_columns = ["lookback", "pred_len", "open", "high", "low", "close", "volume", "amount"]
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Forecast file contains invalid numeric values")

    unsupported = normalized.loc[normalized["pred_len"] != 1, "pred_len"].unique()
    if len(unsupported) > 0:
        raise EvaluationError("Forecast evaluation only supports pred_len=1")

    return normalized


def _load_clean_by_timeframe(manifest_payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    clean_by_timeframe: dict[str, pd.DataFrame] = {}
    for dataset in manifest_payload.get("datasets", []):
        timeframe = dataset["timeframe"]
        clean_path = Path(dataset["clean_path"])
        if not clean_path.exists():
            raise EvaluationError(f"Clean candle file does not exist: {clean_path}")
        clean_by_timeframe[timeframe] = _normalize_clean(
            pd.read_csv(clean_path),
            timeframe=timeframe,
        )

    if not clean_by_timeframe:
        raise EvaluationError("Manifest contains no datasets to evaluate")
    return clean_by_timeframe


def _evaluate_forecast_row(
    *,
    forecast_row: pd.Series,
    clean_by_timeframe: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    timeframe = str(forecast_row["timeframe"])
    if timeframe not in clean_by_timeframe:
        raise EvaluationError(f"Manifest has no clean dataset for forecast timeframe: {timeframe}")

    clean = clean_by_timeframe[timeframe]
    input_end_timestamp = pd.Timestamp(forecast_row["input_end_timestamp"])
    forecast_timestamp = pd.Timestamp(forecast_row["forecast_timestamp"])
    current = _lookup_candle(
        clean,
        timeframe=timeframe,
        timestamp=input_end_timestamp,
        label="input end",
    )
    target = _lookup_candle(
        clean,
        timeframe=timeframe,
        timestamp=forecast_timestamp,
        label="forecast",
    )

    current_close = float(current["close"])
    target_close = float(target["close"])
    kronos_close = float(forecast_row["close"])
    kronos_error = kronos_close - target_close
    naive_close = current_close
    naive_error = naive_close - target_close
    actual_return = _safe_return(target_close, current_close)
    forecasted_return = _safe_return(kronos_close, current_close)
    naive_return = 0.0

    return {
        "run_id": str(forecast_row["run_id"]),
        "forecast_created_at": _format_timestamp(pd.Timestamp(forecast_row["forecast_created_at"])),
        "exchange": str(forecast_row["exchange"]),
        "symbol": str(forecast_row["symbol"]),
        "timeframe": timeframe,
        "model_name": str(forecast_row["model_name"]),
        "lookback": int(forecast_row["lookback"]),
        "pred_len": int(forecast_row["pred_len"]),
        "input_end_timestamp": _format_timestamp(input_end_timestamp),
        "forecast_timestamp": _format_timestamp(forecast_timestamp),
        "current_close": current_close,
        "target_close": target_close,
        "kronos_close": kronos_close,
        "kronos_close_error": kronos_error,
        "kronos_absolute_error": abs(kronos_error),
        "kronos_squared_error": kronos_error**2,
        "actual_return": actual_return,
        "forecasted_return": forecasted_return,
        "kronos_direction_hit": bool(np.sign(forecasted_return) == np.sign(actual_return)),
        "naive_close": naive_close,
        "naive_close_error": naive_error,
        "naive_absolute_error": abs(naive_error),
        "naive_squared_error": naive_error**2,
        "naive_return": naive_return,
        "naive_direction_hit": bool(np.sign(naive_return) == np.sign(actual_return)),
    }


def _lookup_candle(
    clean: pd.DataFrame,
    *,
    timeframe: str,
    timestamp: pd.Timestamp,
    label: str,
) -> pd.Series:
    matches = clean.loc[clean["timestamp"] == timestamp]
    if matches.empty:
        raise EvaluationError(
            f"No realized {timeframe} candle found for {label} timestamp: {_format_timestamp(timestamp)}"
        )
    if len(matches) > 1:
        raise EvaluationError(
            f"Multiple realized {timeframe} candles found for {label} timestamp: {_format_timestamp(timestamp)}"
        )
    return matches.iloc[0]


def _safe_return(value: float, base: float) -> float:
    if base == 0:
        raise EvaluationError("Cannot compute return from zero current close")
    return float((value - base) / base)


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
