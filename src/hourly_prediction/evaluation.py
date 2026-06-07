from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.kronos_runner import (
    DEFAULT_DEVICE,
    DEFAULT_LOOKBACK,
    DEFAULT_PRED_LEN,
    PredictorLoader,
    load_kronos_predictor,
    resolve_manifest_path,
)
from hourly_prediction.kronos_runtime import (
    DEFAULT_KRONOS_MODEL,
    DEFAULT_KRONOS_TOKENIZER,
)
from hourly_prediction.validation import timeframe_delta


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
WALK_FORWARD_EVALUATION_COLUMNS = [
    "run_id",
    "evaluation_created_at",
    "exchange",
    "symbol",
    "timeframe",
    "model_name",
    "top_p",
    "sample_count",
    "window_selection",
    "input_transform",
    "lookback",
    "pred_len",
    "window_number",
    "input_start_timestamp",
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
    "sma_close",
    "sma_close_error",
    "sma_absolute_error",
    "sma_squared_error",
    "sma_return",
    "sma_direction_hit",
]
WALK_FORWARD_SUMMARY_REQUIRED_COLUMNS = [
    "model_name",
    "top_p",
    "sample_count",
    "window_selection",
    "input_transform",
    "timeframe",
    "kronos_absolute_error",
    "kronos_squared_error",
    "naive_absolute_error",
    "naive_squared_error",
    "sma_absolute_error",
    "sma_squared_error",
    "kronos_direction_hit",
    "naive_direction_hit",
    "sma_direction_hit",
    "actual_return",
    "forecasted_return",
]
WALK_FORWARD_SUMMARY_COLUMNS = [
    "model_name",
    "top_p",
    "sample_count",
    "window_selection",
    "input_transform",
    "timeframe",
    "rows",
    "kronos_mae",
    "kronos_rmse",
    "naive_mae",
    "naive_rmse",
    "sma_mae",
    "sma_rmse",
    "kronos_directional_accuracy",
    "naive_directional_accuracy",
    "sma_directional_accuracy",
    "average_actual_return",
    "average_forecasted_return",
]
WALK_FORWARD_DIAGNOSTIC_REQUIRED_COLUMNS = [
    "model_name",
    "top_p",
    "sample_count",
    "window_selection",
    "input_transform",
    "timeframe",
    "kronos_close_error",
    "kronos_absolute_error",
    "naive_close_error",
    "naive_absolute_error",
    "sma_close_error",
    "sma_absolute_error",
    "actual_return",
    "forecasted_return",
]
_DIRECTION_LABELS = ("up", "down", "flat")
WALK_FORWARD_DIAGNOSTIC_COLUMNS = [
    "model_name",
    "top_p",
    "sample_count",
    "window_selection",
    "input_transform",
    "timeframe",
    "rows",
    "random_seed",
    "kronos_mean_signed_error",
    "naive_mean_signed_error",
    "sma_mean_signed_error",
    "kronos_median_absolute_error",
    "naive_median_absolute_error",
    "sma_median_absolute_error",
    "kronos_error_std",
    "naive_error_std",
    "sma_error_std",
    "kronos_vs_naive_mae_delta",
    "kronos_vs_naive_mae_ratio",
    "kronos_vs_sma_mae_delta",
    "kronos_vs_sma_mae_ratio",
    "average_actual_return",
    "average_forecasted_return",
    "random_directional_accuracy",
    *[
        f"kronos_actual_{actual}_pred_{predicted}"
        for actual in _DIRECTION_LABELS
        for predicted in _DIRECTION_LABELS
    ],
]
WALK_FORWARD_COMPARISON_KEY_COLUMNS = [
    "model_name",
    "top_p",
    "sample_count",
    "window_selection",
    "input_transform",
    "timeframe",
]
WALK_FORWARD_COMPARISON_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "rows",
    "kronos_mae",
    "kronos_rmse",
    "naive_mae",
    "naive_rmse",
    "sma_mae",
    "sma_rmse",
    "kronos_directional_accuracy",
    "naive_directional_accuracy",
    "sma_directional_accuracy",
    "random_directional_accuracy",
    "kronos_mean_signed_error",
    "naive_mean_signed_error",
    "sma_mean_signed_error",
    "kronos_vs_naive_mae_delta",
    "kronos_vs_naive_mae_ratio",
    "kronos_vs_sma_mae_delta",
    "kronos_vs_sma_mae_ratio",
    "average_actual_return",
    "average_forecasted_return",
    "beats_naive_mae",
    "beats_sma_mae",
]
WALK_FORWARD_REGIME_REQUIRED_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "forecast_timestamp",
    "kronos_absolute_error",
    "kronos_squared_error",
    "naive_absolute_error",
    "naive_squared_error",
    "sma_absolute_error",
    "sma_squared_error",
    "kronos_direction_hit",
    "naive_direction_hit",
    "sma_direction_hit",
    "actual_return",
    "forecasted_return",
]
WALK_FORWARD_REGIME_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "return_regime",
    "rows",
    "average_absolute_actual_return",
    "kronos_mae",
    "kronos_rmse",
    "naive_mae",
    "naive_rmse",
    "sma_mae",
    "sma_rmse",
    "kronos_vs_naive_mae_ratio",
    "kronos_directional_accuracy",
    "naive_directional_accuracy",
    "sma_directional_accuracy",
    "average_actual_return",
    "average_forecasted_return",
    "forecast_actual_return_correlation",
]
TARGET_FORMULATION_REQUIRED_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "kronos_close_error",
    "kronos_absolute_error",
    "kronos_squared_error",
    "naive_close_error",
    "naive_absolute_error",
    "naive_squared_error",
    "sma_close_error",
    "sma_absolute_error",
    "sma_squared_error",
    "actual_return",
    "forecasted_return",
    "naive_return",
    "sma_return",
]
TARGET_FORMULATION_BASE_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "rows",
    "kronos_close_mae",
    "kronos_close_rmse",
    "naive_close_mae",
    "naive_close_rmse",
    "sma_close_mae",
    "sma_close_rmse",
    "kronos_return_mae",
    "kronos_return_rmse",
    "naive_return_mae",
    "naive_return_rmse",
    "sma_return_mae",
    "sma_return_rmse",
    "kronos_mean_signed_close_error",
    "kronos_median_signed_close_error",
    "naive_mean_signed_close_error",
    "naive_median_signed_close_error",
    "sma_mean_signed_close_error",
    "sma_median_signed_close_error",
    "kronos_mean_signed_return_error",
    "kronos_median_signed_return_error",
    "naive_mean_signed_return_error",
    "naive_median_signed_return_error",
    "sma_mean_signed_return_error",
    "sma_median_signed_return_error",
    "forecast_actual_return_correlation",
    "kronos_beats_naive_close_mae",
    "kronos_beats_naive_return_mae",
]
DEFAULT_MAX_WALK_FORWARD_WINDOWS = 20
DEFAULT_SMA_WINDOW = 20
DEFAULT_RANDOM_BASELINE_SEED = 42
DEFAULT_WINDOW_SELECTION = "recent"
SUPPORTED_WINDOW_SELECTIONS = ("recent", "even")
DEFAULT_INPUT_TRANSFORM = "raw"
SUPPORTED_INPUT_TRANSFORMS = ("raw", "relative")
DEFAULT_RETURN_REGIME_BUCKETS = 3
DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS = (0, 5, 10, 25)
TARGET_FORMULATION_COLUMNS = [
    *TARGET_FORMULATION_BASE_COLUMNS,
    *[
        column
        for threshold in DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS
        for column in (
            f"threshold_{threshold}_bps_rows",
            f"threshold_{threshold}_bps_kronos_directional_accuracy",
            f"threshold_{threshold}_bps_naive_directional_accuracy",
            f"threshold_{threshold}_bps_sma_directional_accuracy",
            f"kronos_beats_naive_threshold_{threshold}_bps_direction",
        )
    ],
    "kronos_beats_naive_any_threshold_direction",
]
FORECAST_CALIBRATION_REQUIRED_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "forecast_timestamp",
    "current_close",
    "actual_return",
    "forecasted_return",
    "naive_return",
    "sma_return",
]
FORECAST_CALIBRATION_BASE_COLUMNS = [
    *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
    "train_fraction",
    "train_rows",
    "test_rows",
    "train_start_timestamp",
    "train_end_timestamp",
    "test_start_timestamp",
    "test_end_timestamp",
    "bias_correction",
    "linear_alpha",
    "linear_beta",
    "linear_degenerate",
    "test_forecast_actual_return_correlation",
    "uncalibrated_return_mae",
    "uncalibrated_return_rmse",
    "uncalibrated_close_mae",
    "uncalibrated_close_rmse",
    "bias_return_mae",
    "bias_return_rmse",
    "bias_close_mae",
    "bias_close_rmse",
    "linear_return_mae",
    "linear_return_rmse",
    "linear_close_mae",
    "linear_close_rmse",
    "naive_return_mae",
    "naive_return_rmse",
    "naive_close_mae",
    "naive_close_rmse",
    "sma_return_mae",
    "sma_return_rmse",
    "sma_close_mae",
    "sma_close_rmse",
    "uncalibrated_beats_naive_return_mae",
    "uncalibrated_beats_naive_close_mae",
    "bias_beats_naive_return_mae",
    "bias_beats_naive_close_mae",
    "linear_beats_naive_return_mae",
    "linear_beats_naive_close_mae",
]
FORECAST_CALIBRATION_COLUMNS = [
    *FORECAST_CALIBRATION_BASE_COLUMNS,
    *[
        column
        for threshold in DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS
        for column in (
            f"threshold_{threshold}_bps_rows",
            f"threshold_{threshold}_bps_uncalibrated_directional_accuracy",
            f"threshold_{threshold}_bps_bias_directional_accuracy",
            f"threshold_{threshold}_bps_linear_directional_accuracy",
            f"threshold_{threshold}_bps_naive_directional_accuracy",
            f"threshold_{threshold}_bps_sma_directional_accuracy",
            f"uncalibrated_beats_naive_threshold_{threshold}_bps_direction",
            f"bias_beats_naive_threshold_{threshold}_bps_direction",
            f"linear_beats_naive_threshold_{threshold}_bps_direction",
        )
    ],
    "uncalibrated_beats_naive_any_threshold_direction",
    "bias_beats_naive_any_threshold_direction",
    "linear_beats_naive_any_threshold_direction",
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


@dataclass(frozen=True)
class WalkForwardEvaluationRun:
    manifest_path: Path
    output_path: Path
    rows: int


@dataclass(frozen=True)
class WalkForwardSummaryRun:
    metrics_path: Path
    output_path: Path
    rows: int


@dataclass(frozen=True)
class WalkForwardDiagnosticRun:
    metrics_path: Path
    output_path: Path
    rows: int


@dataclass(frozen=True)
class WalkForwardComparisonRun:
    summary_paths: tuple[Path, ...]
    diagnostic_paths: tuple[Path, ...]
    output_path: Path
    rows: int


@dataclass(frozen=True)
class WalkForwardRegimeRun:
    metrics_paths: tuple[Path, ...]
    output_path: Path
    rows: int


@dataclass(frozen=True)
class TargetFormulationRun:
    metrics_paths: tuple[Path, ...]
    output_path: Path
    rows: int


@dataclass(frozen=True)
class ForecastCalibrationRun:
    metrics_paths: tuple[Path, ...]
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


def evaluate_kronos_walk_forward(
    *,
    manifest: str | Path,
    manifest_dir: Path,
    output_dir: Path,
    kronos_repo_path: Path,
    device: str = DEFAULT_DEVICE,
    lookback: int = DEFAULT_LOOKBACK,
    max_windows: int = DEFAULT_MAX_WALK_FORWARD_WINDOWS,
    sma_window: int = DEFAULT_SMA_WINDOW,
    pred_len: int = DEFAULT_PRED_LEN,
    model_name: str = DEFAULT_KRONOS_MODEL,
    tokenizer_name: str = DEFAULT_KRONOS_TOKENIZER,
    top_p: float = 0.9,
    sample_count: int = 1,
    window_selection: str = DEFAULT_WINDOW_SELECTION,
    input_transform: str = DEFAULT_INPUT_TRANSFORM,
    now: pd.Timestamp | None = None,
    predictor_loader: PredictorLoader = None,
) -> WalkForwardEvaluationRun:
    if pred_len != 1:
        raise EvaluationError("Walk-forward evaluation only supports pred_len=1")
    if lookback <= 0:
        raise EvaluationError("lookback must be positive")
    if max_windows <= 0:
        raise EvaluationError("max_windows must be positive")
    if sma_window <= 0:
        raise EvaluationError("sma_window must be positive")
    if top_p <= 0 or top_p > 1:
        raise EvaluationError("top_p must be greater than 0 and at most 1")
    if sample_count <= 0:
        raise EvaluationError("sample_count must be positive")
    if window_selection not in SUPPORTED_WINDOW_SELECTIONS:
        raise EvaluationError(
            "window_selection must be one of: " + ", ".join(SUPPORTED_WINDOW_SELECTIONS)
        )
    if input_transform not in SUPPORTED_INPUT_TRANSFORMS:
        raise EvaluationError(
            "input_transform must be one of: " + ", ".join(SUPPORTED_INPUT_TRANSFORMS)
        )
    if lookback < sma_window:
        raise EvaluationError("lookback must be at least sma_window")

    manifest_path = resolve_manifest_path(manifest, manifest_dir=manifest_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_by_timeframe = _load_clean_by_timeframe(payload)
    evaluation_created_at = _format_timestamp(_normalize_timestamp(now))
    predictor = (predictor_loader or load_kronos_predictor)(
        kronos_repo_path=kronos_repo_path,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
        max_context=lookback,
    )

    rows: list[dict[str, Any]] = []
    for dataset in payload.get("datasets", []):
        timeframe = dataset["timeframe"]
        clean = clean_by_timeframe[timeframe]
        windows = build_evaluation_windows(clean, timeframe=timeframe, lookback=lookback)
        selected = _select_walk_forward_windows(
            windows,
            max_windows=max_windows,
            window_selection=window_selection,
        )
        for window_number, window in enumerate(selected, start=1):
            rows.append(
                _evaluate_walk_forward_window(
                    window=window,
                    manifest=payload,
                    predictor=predictor,
                    evaluation_created_at=evaluation_created_at,
                    model_name=model_name,
                    top_p=top_p,
                    sample_count=sample_count,
                    window_selection=window_selection,
                    input_transform=input_transform,
                    lookback=lookback,
                    pred_len=pred_len,
                    window_number=window_number,
                    sma_window=sma_window,
                )
            )

    if not rows:
        raise EvaluationError("Walk-forward evaluation produced no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{payload['run_id']}_{_model_slug(model_name)}_"
        f"{_sampling_slug(top_p=top_p, sample_count=sample_count)}_"
        f"{window_selection}_{input_transform}_walk_forward_metrics.csv"
    )
    pd.DataFrame(rows, columns=WALK_FORWARD_EVALUATION_COLUMNS).to_csv(output_path, index=False)
    return WalkForwardEvaluationRun(
        manifest_path=manifest_path,
        output_path=output_path,
        rows=len(rows),
    )


def summarize_walk_forward_metrics(
    *,
    metrics: str | Path,
    output_dir: Path,
) -> WalkForwardSummaryRun:
    metrics_path = Path(metrics)
    if not metrics_path.exists():
        raise EvaluationError(f"Walk-forward metrics file does not exist: {metrics_path}")

    metric_rows = _normalize_walk_forward_metrics(pd.read_csv(metrics_path))
    rows: list[dict[str, Any]] = []
    for (
        model_name,
        top_p,
        sample_count,
        window_selection,
        input_transform,
        timeframe,
    ), group in metric_rows.groupby(
        WALK_FORWARD_COMPARISON_KEY_COLUMNS,
        sort=True,
    ):
        rows.append(
            {
                "model_name": model_name,
                "top_p": float(top_p),
                "sample_count": int(sample_count),
                "window_selection": window_selection,
                "input_transform": input_transform,
                "timeframe": timeframe,
                "rows": int(len(group)),
                "kronos_mae": float(group["kronos_absolute_error"].mean()),
                "kronos_rmse": float(np.sqrt(group["kronos_squared_error"].mean())),
                "naive_mae": float(group["naive_absolute_error"].mean()),
                "naive_rmse": float(np.sqrt(group["naive_squared_error"].mean())),
                "sma_mae": float(group["sma_absolute_error"].mean()),
                "sma_rmse": float(np.sqrt(group["sma_squared_error"].mean())),
                "kronos_directional_accuracy": float(group["kronos_direction_hit"].mean()),
                "naive_directional_accuracy": float(group["naive_direction_hit"].mean()),
                "sma_directional_accuracy": float(group["sma_direction_hit"].mean()),
                "average_actual_return": float(group["actual_return"].mean()),
                "average_forecasted_return": float(group["forecasted_return"].mean()),
            }
        )

    if not rows:
        raise EvaluationError("Walk-forward metrics file contains no rows to summarize")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _summary_filename(metrics_path)
    pd.DataFrame(rows, columns=WALK_FORWARD_SUMMARY_COLUMNS).to_csv(output_path, index=False)
    return WalkForwardSummaryRun(
        metrics_path=metrics_path,
        output_path=output_path,
        rows=len(rows),
    )


def diagnose_walk_forward_metrics(
    *,
    metrics: str | Path,
    output_dir: Path,
    random_seed: int = DEFAULT_RANDOM_BASELINE_SEED,
) -> WalkForwardDiagnosticRun:
    metrics_path = Path(metrics)
    if not metrics_path.exists():
        raise EvaluationError(f"Walk-forward metrics file does not exist: {metrics_path}")

    metric_rows = _normalize_walk_forward_diagnostics(pd.read_csv(metrics_path))
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    for (
        model_name,
        top_p,
        sample_count,
        window_selection,
        input_transform,
        timeframe,
    ), group in metric_rows.groupby(
        WALK_FORWARD_COMPARISON_KEY_COLUMNS,
        sort=True,
    ):
        rows.append(
            _diagnose_timeframe_group(
                model_name=model_name,
                top_p=float(top_p),
                sample_count=int(sample_count),
                window_selection=window_selection,
                input_transform=input_transform,
                timeframe=timeframe,
                group=group,
                random_seed=random_seed,
                rng=rng,
            )
        )

    if not rows:
        raise EvaluationError("Walk-forward metrics file contains no rows to diagnose")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _diagnostic_filename(metrics_path)
    pd.DataFrame(rows, columns=WALK_FORWARD_DIAGNOSTIC_COLUMNS).to_csv(output_path, index=False)
    return WalkForwardDiagnosticRun(
        metrics_path=metrics_path,
        output_path=output_path,
        rows=len(rows),
    )


def compare_walk_forward_reports(
    *,
    summaries: list[str | Path],
    diagnostics: list[str | Path],
    output_dir: Path,
    output_name: str = "walk_forward_model_comparison.csv",
) -> WalkForwardComparisonRun:
    if not summaries:
        raise EvaluationError("At least one walk-forward summary file is required")
    if not diagnostics:
        raise EvaluationError("At least one walk-forward diagnostics file is required")

    summary_paths = tuple(Path(path) for path in summaries)
    diagnostic_paths = tuple(Path(path) for path in diagnostics)
    for path in (*summary_paths, *diagnostic_paths):
        if not path.exists():
            raise EvaluationError(f"Walk-forward report file does not exist: {path}")

    summary_rows = _normalize_walk_forward_summary_reports(
        pd.concat((pd.read_csv(path) for path in summary_paths), ignore_index=True)
    )
    diagnostic_rows = _normalize_walk_forward_diagnostic_reports(
        pd.concat((pd.read_csv(path) for path in diagnostic_paths), ignore_index=True)
    )
    comparison = summary_rows.merge(
        diagnostic_rows,
        on=WALK_FORWARD_COMPARISON_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_diagnostic"),
    )
    if comparison["random_directional_accuracy"].isna().any():
        raise EvaluationError("Walk-forward comparison missing matching diagnostics")

    rows = comparison[WALK_FORWARD_COMPARISON_COLUMNS[:-2]].copy()
    rows["beats_naive_mae"] = rows["kronos_mae"] < rows["naive_mae"]
    rows["beats_sma_mae"] = rows["kronos_mae"] < rows["sma_mae"]
    rows = rows.sort_values(WALK_FORWARD_COMPARISON_KEY_COLUMNS).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    rows.to_csv(output_path, index=False, columns=WALK_FORWARD_COMPARISON_COLUMNS)
    return WalkForwardComparisonRun(
        summary_paths=summary_paths,
        diagnostic_paths=diagnostic_paths,
        output_path=output_path,
        rows=len(rows),
    )


def analyze_walk_forward_regimes(
    *,
    metrics: list[str | Path],
    output_dir: Path,
    bucket_count: int = DEFAULT_RETURN_REGIME_BUCKETS,
    output_name: str = "walk_forward_regime_diagnostics.csv",
) -> WalkForwardRegimeRun:
    if not metrics:
        raise EvaluationError("At least one walk-forward metrics file is required")
    if bucket_count <= 0:
        raise EvaluationError("bucket_count must be positive")

    metrics_paths = tuple(Path(path) for path in metrics)
    for path in metrics_paths:
        if not path.exists():
            raise EvaluationError(f"Walk-forward metrics file does not exist: {path}")

    metric_rows = _normalize_walk_forward_regime_metrics(
        pd.concat((pd.read_csv(path) for path in metrics_paths), ignore_index=True)
    )
    metric_rows = _add_return_regime_buckets(metric_rows, bucket_count=bucket_count)
    rows: list[dict[str, Any]] = []
    for (
        model_name,
        top_p,
        sample_count,
        window_selection,
        input_transform,
        timeframe,
        return_regime,
    ), group in metric_rows.groupby(
        [*WALK_FORWARD_COMPARISON_KEY_COLUMNS, "return_regime"],
        sort=True,
    ):
        kronos_mae = float(group["kronos_absolute_error"].mean())
        naive_mae = float(group["naive_absolute_error"].mean())
        rows.append(
            {
                "model_name": model_name,
                "top_p": float(top_p),
                "sample_count": int(sample_count),
                "window_selection": window_selection,
                "input_transform": input_transform,
                "timeframe": timeframe,
                "return_regime": return_regime,
                "rows": int(len(group)),
                "average_absolute_actual_return": float(
                    group["absolute_actual_return"].mean()
                ),
                "kronos_mae": kronos_mae,
                "kronos_rmse": float(np.sqrt(group["kronos_squared_error"].mean())),
                "naive_mae": naive_mae,
                "naive_rmse": float(np.sqrt(group["naive_squared_error"].mean())),
                "sma_mae": float(group["sma_absolute_error"].mean()),
                "sma_rmse": float(np.sqrt(group["sma_squared_error"].mean())),
                "kronos_vs_naive_mae_ratio": _safe_ratio(kronos_mae, naive_mae),
                "kronos_directional_accuracy": float(group["kronos_direction_hit"].mean()),
                "naive_directional_accuracy": float(group["naive_direction_hit"].mean()),
                "sma_directional_accuracy": float(group["sma_direction_hit"].mean()),
                "average_actual_return": float(group["actual_return"].mean()),
                "average_forecasted_return": float(group["forecasted_return"].mean()),
                "forecast_actual_return_correlation": _safe_correlation(
                    group["forecasted_return"],
                    group["actual_return"],
                ),
            }
        )

    if not rows:
        raise EvaluationError("Walk-forward regime analysis produced no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    pd.DataFrame(rows, columns=WALK_FORWARD_REGIME_COLUMNS).to_csv(output_path, index=False)
    return WalkForwardRegimeRun(
        metrics_paths=metrics_paths,
        output_path=output_path,
        rows=len(rows),
    )


def analyze_target_formulation(
    *,
    metrics: list[str | Path],
    output_dir: Path,
    thresholds_bps: list[int] | tuple[int, ...] = DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS,
    output_name: str = "walk_forward_target_formulation.csv",
) -> TargetFormulationRun:
    if not metrics:
        raise EvaluationError("At least one walk-forward metrics file is required")
    thresholds = tuple(int(threshold) for threshold in thresholds_bps)
    if not thresholds:
        raise EvaluationError("At least one target formulation threshold is required")
    if any(threshold < 0 for threshold in thresholds):
        raise EvaluationError("Target formulation thresholds must be non-negative")
    if len(set(thresholds)) != len(thresholds):
        raise EvaluationError("Target formulation thresholds must be unique")

    metrics_paths = tuple(Path(path) for path in metrics)
    for path in metrics_paths:
        if not path.exists():
            raise EvaluationError(f"Walk-forward metrics file does not exist: {path}")

    metric_rows = _normalize_target_formulation_metrics(
        pd.concat((pd.read_csv(path) for path in metrics_paths), ignore_index=True)
    )
    rows: list[dict[str, Any]] = []
    for (
        model_name,
        top_p,
        sample_count,
        window_selection,
        input_transform,
        timeframe,
    ), group in metric_rows.groupby(
        WALK_FORWARD_COMPARISON_KEY_COLUMNS,
        sort=True,
    ):
        rows.append(
            _analyze_target_formulation_group(
                model_name=model_name,
                top_p=float(top_p),
                sample_count=int(sample_count),
                window_selection=window_selection,
                input_transform=input_transform,
                timeframe=timeframe,
                group=group,
                thresholds_bps=thresholds,
            )
        )

    if not rows:
        raise EvaluationError("Target formulation analysis produced no rows")

    columns = _target_formulation_columns(thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    return TargetFormulationRun(
        metrics_paths=metrics_paths,
        output_path=output_path,
        rows=len(rows),
    )


def analyze_forecast_calibration(
    *,
    metrics: list[str | Path],
    output_dir: Path,
    train_fraction: float = 0.7,
    thresholds_bps: list[int] | tuple[int, ...] = DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS,
    output_name: str = "walk_forward_forecast_calibration.csv",
) -> ForecastCalibrationRun:
    if not metrics:
        raise EvaluationError("At least one walk-forward metrics file is required")
    if train_fraction <= 0 or train_fraction >= 1:
        raise EvaluationError("train_fraction must be greater than 0 and less than 1")
    thresholds = tuple(int(threshold) for threshold in thresholds_bps)
    if not thresholds:
        raise EvaluationError("At least one forecast calibration threshold is required")
    if any(threshold < 0 for threshold in thresholds):
        raise EvaluationError("Forecast calibration thresholds must be non-negative")
    if len(set(thresholds)) != len(thresholds):
        raise EvaluationError("Forecast calibration thresholds must be unique")

    metrics_paths = tuple(Path(path) for path in metrics)
    for path in metrics_paths:
        if not path.exists():
            raise EvaluationError(f"Walk-forward metrics file does not exist: {path}")

    metric_rows = _normalize_forecast_calibration_metrics(
        pd.concat((pd.read_csv(path) for path in metrics_paths), ignore_index=True)
    )
    rows: list[dict[str, Any]] = []
    for (
        model_name,
        top_p,
        sample_count,
        window_selection,
        input_transform,
        timeframe,
    ), group in metric_rows.groupby(
        WALK_FORWARD_COMPARISON_KEY_COLUMNS,
        sort=True,
    ):
        rows.append(
            _analyze_forecast_calibration_group(
                model_name=model_name,
                top_p=float(top_p),
                sample_count=int(sample_count),
                window_selection=window_selection,
                input_transform=input_transform,
                timeframe=timeframe,
                group=group,
                train_fraction=train_fraction,
                thresholds_bps=thresholds,
            )
        )

    if not rows:
        raise EvaluationError("Forecast calibration analysis produced no rows")

    columns = _forecast_calibration_columns(thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    return ForecastCalibrationRun(
        metrics_paths=metrics_paths,
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


def _normalize_walk_forward_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = _with_default_input_transform(metrics)
    missing = [
        column
        for column in WALK_FORWARD_SUMMARY_REQUIRED_COLUMNS
        if column not in metrics.columns
    ]
    if missing:
        raise EvaluationError(
            "Walk-forward metrics missing required column(s): " + ", ".join(missing)
        )

    normalized = metrics.copy()
    numeric_columns = [
        "top_p",
        "sample_count",
        "kronos_absolute_error",
        "kronos_squared_error",
        "naive_absolute_error",
        "naive_squared_error",
        "sma_absolute_error",
        "sma_squared_error",
        "actual_return",
        "forecasted_return",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Walk-forward metrics contain invalid numeric values")
    if (normalized["top_p"] <= 0).any() or (normalized["top_p"] > 1).any():
        raise EvaluationError("Walk-forward metrics contain invalid top_p values")
    if (normalized["sample_count"] <= 0).any():
        raise EvaluationError("Walk-forward metrics contain invalid sample_count values")

    for column in ("kronos_direction_hit", "naive_direction_hit", "sma_direction_hit"):
        normalized[column] = _coerce_boolean_series(normalized[column], column=column)

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    return normalized


def _normalize_walk_forward_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = _with_default_input_transform(metrics)
    missing = [
        column
        for column in WALK_FORWARD_DIAGNOSTIC_REQUIRED_COLUMNS
        if column not in metrics.columns
    ]
    if missing:
        raise EvaluationError(
            "Walk-forward diagnostics missing required column(s): " + ", ".join(missing)
        )

    normalized = metrics.copy()
    numeric_columns = [
        "top_p",
        "sample_count",
        "kronos_close_error",
        "kronos_absolute_error",
        "naive_close_error",
        "naive_absolute_error",
        "sma_close_error",
        "sma_absolute_error",
        "actual_return",
        "forecasted_return",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Walk-forward diagnostics contain invalid numeric values")
    if (normalized["top_p"] <= 0).any() or (normalized["top_p"] > 1).any():
        raise EvaluationError("Walk-forward diagnostics contain invalid top_p values")
    if (normalized["sample_count"] <= 0).any():
        raise EvaluationError("Walk-forward diagnostics contain invalid sample_count values")

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    return normalized


def _normalize_walk_forward_summary_reports(reports: pd.DataFrame) -> pd.DataFrame:
    reports = _with_default_input_transform(reports)
    missing = [
        column
        for column in WALK_FORWARD_SUMMARY_COLUMNS
        if column not in reports.columns
    ]
    if missing:
        raise EvaluationError(
            "Walk-forward summary report missing required column(s): " + ", ".join(missing)
        )

    normalized = reports[WALK_FORWARD_SUMMARY_COLUMNS].copy()
    numeric_columns = [
        column
        for column in WALK_FORWARD_SUMMARY_COLUMNS
        if column not in ("model_name", "window_selection", "input_transform", "timeframe")
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Walk-forward summary report contains invalid numeric values")

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    normalized["rows"] = normalized["rows"].astype(int)
    return normalized


def _normalize_walk_forward_diagnostic_reports(reports: pd.DataFrame) -> pd.DataFrame:
    reports = _with_default_input_transform(reports)
    required = [
        *WALK_FORWARD_COMPARISON_KEY_COLUMNS,
        "random_directional_accuracy",
        "kronos_mean_signed_error",
        "naive_mean_signed_error",
        "sma_mean_signed_error",
        "kronos_vs_naive_mae_delta",
        "kronos_vs_naive_mae_ratio",
        "kronos_vs_sma_mae_delta",
        "kronos_vs_sma_mae_ratio",
    ]
    missing = [column for column in required if column not in reports.columns]
    if missing:
        raise EvaluationError(
            "Walk-forward diagnostic report missing required column(s): " + ", ".join(missing)
        )

    normalized = reports[required].copy()
    numeric_columns = [
        column
        for column in required
        if column not in ("model_name", "window_selection", "input_transform", "timeframe")
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Walk-forward diagnostic report contains invalid numeric values")

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    return normalized


def _normalize_walk_forward_regime_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = _with_default_input_transform(metrics)
    missing = [
        column
        for column in WALK_FORWARD_REGIME_REQUIRED_COLUMNS
        if column not in metrics.columns
    ]
    if missing:
        raise EvaluationError(
            "Walk-forward regime metrics missing required column(s): " + ", ".join(missing)
        )

    normalized = metrics[WALK_FORWARD_REGIME_REQUIRED_COLUMNS].copy()
    numeric_columns = [
        "top_p",
        "sample_count",
        "kronos_absolute_error",
        "kronos_squared_error",
        "naive_absolute_error",
        "naive_squared_error",
        "sma_absolute_error",
        "sma_squared_error",
        "actual_return",
        "forecasted_return",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Walk-forward regime metrics contain invalid numeric values")
    if (normalized["top_p"] <= 0).any() or (normalized["top_p"] > 1).any():
        raise EvaluationError("Walk-forward regime metrics contain invalid top_p values")
    if (normalized["sample_count"] <= 0).any():
        raise EvaluationError("Walk-forward regime metrics contain invalid sample_count values")

    normalized["forecast_timestamp"] = pd.to_datetime(
        normalized["forecast_timestamp"],
        utc=True,
        errors="coerce",
    )
    if normalized["forecast_timestamp"].isna().any():
        raise EvaluationError("Walk-forward regime metrics contain invalid forecast timestamps")

    for column in ("kronos_direction_hit", "naive_direction_hit", "sma_direction_hit"):
        normalized[column] = _coerce_boolean_series(normalized[column], column=column)

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    normalized["absolute_actual_return"] = normalized["actual_return"].abs()
    return normalized


def _normalize_target_formulation_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = _with_default_input_transform(metrics)
    missing = [
        column
        for column in TARGET_FORMULATION_REQUIRED_COLUMNS
        if column not in metrics.columns
    ]
    if missing:
        raise EvaluationError(
            "Target formulation metrics missing required column(s): " + ", ".join(missing)
        )

    normalized = metrics[TARGET_FORMULATION_REQUIRED_COLUMNS].copy()
    numeric_columns = [
        "top_p",
        "sample_count",
        "kronos_close_error",
        "kronos_absolute_error",
        "kronos_squared_error",
        "naive_close_error",
        "naive_absolute_error",
        "naive_squared_error",
        "sma_close_error",
        "sma_absolute_error",
        "sma_squared_error",
        "actual_return",
        "forecasted_return",
        "naive_return",
        "sma_return",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Target formulation metrics contain invalid numeric values")
    if (normalized["top_p"] <= 0).any() or (normalized["top_p"] > 1).any():
        raise EvaluationError("Target formulation metrics contain invalid top_p values")
    if (normalized["sample_count"] <= 0).any():
        raise EvaluationError("Target formulation metrics contain invalid sample_count values")

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    return normalized


def _normalize_forecast_calibration_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = _with_default_input_transform(metrics)
    missing = [
        column
        for column in FORECAST_CALIBRATION_REQUIRED_COLUMNS
        if column not in metrics.columns
    ]
    if missing:
        raise EvaluationError(
            "Forecast calibration metrics missing required column(s): " + ", ".join(missing)
        )

    normalized = metrics[FORECAST_CALIBRATION_REQUIRED_COLUMNS].copy()
    numeric_columns = [
        "top_p",
        "sample_count",
        "current_close",
        "actual_return",
        "forecasted_return",
        "naive_return",
        "sma_return",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[numeric_columns].isna().any().any():
        raise EvaluationError("Forecast calibration metrics contain invalid numeric values")
    if (normalized["top_p"] <= 0).any() or (normalized["top_p"] > 1).any():
        raise EvaluationError("Forecast calibration metrics contain invalid top_p values")
    if (normalized["sample_count"] <= 0).any():
        raise EvaluationError("Forecast calibration metrics contain invalid sample_count values")
    if (normalized["current_close"] <= 0).any():
        raise EvaluationError("Forecast calibration metrics contain invalid current_close values")

    normalized["forecast_timestamp"] = pd.to_datetime(
        normalized["forecast_timestamp"],
        utc=True,
        errors="coerce",
    )
    if normalized["forecast_timestamp"].isna().any():
        raise EvaluationError("Forecast calibration metrics contain invalid forecast timestamps")

    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["window_selection"] = normalized["window_selection"].astype(str)
    normalized["input_transform"] = normalized["input_transform"].astype(str)
    normalized["timeframe"] = normalized["timeframe"].astype(str)
    normalized["sample_count"] = normalized["sample_count"].astype(int)
    return normalized


def _with_default_input_transform(rows: pd.DataFrame) -> pd.DataFrame:
    if "input_transform" in rows.columns:
        return rows
    normalized = rows.copy()
    normalized["input_transform"] = DEFAULT_INPUT_TRANSFORM
    return normalized


def _add_return_regime_buckets(metrics: pd.DataFrame, *, bucket_count: int) -> pd.DataFrame:
    normalized = metrics.copy()
    target_columns = ["timeframe", "forecast_timestamp", "actual_return", "absolute_actual_return"]
    targets = normalized[target_columns].drop_duplicates().copy()
    duplicate_targets = targets.duplicated(["timeframe", "forecast_timestamp"], keep=False)
    if duplicate_targets.any():
        raise EvaluationError("Walk-forward regime metrics contain conflicting target returns")

    bucketed_targets: list[pd.DataFrame] = []
    for timeframe, group in targets.groupby("timeframe", sort=True):
        group = group.sort_values(
            ["absolute_actual_return", "forecast_timestamp"],
            kind="mergesort",
        ).reset_index(drop=True)
        regime_count = min(bucket_count, len(group))
        if regime_count == 1:
            group["return_regime"] = "q1_all_abs_return"
        else:
            labels = [f"q{index}_of_{regime_count}_abs_return" for index in range(1, regime_count + 1)]
            ranks = pd.Series(np.arange(1, len(group) + 1), index=group.index)
            group["return_regime"] = pd.qcut(ranks, q=regime_count, labels=labels)
            group["return_regime"] = group["return_regime"].astype(str)
        bucketed_targets.append(group[["timeframe", "forecast_timestamp", "return_regime"]])

    bucketed = pd.concat(bucketed_targets, ignore_index=True)
    return normalized.merge(
        bucketed,
        on=["timeframe", "forecast_timestamp"],
        how="left",
        validate="many_to_one",
    )


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


def _select_walk_forward_windows(
    windows: list[EvaluationWindow],
    *,
    max_windows: int,
    window_selection: str,
) -> list[EvaluationWindow]:
    if len(windows) <= max_windows:
        return list(windows)

    if window_selection == "recent":
        return windows[-max_windows:]

    if window_selection == "even":
        indices = np.rint(np.linspace(0, len(windows) - 1, num=max_windows)).astype(int)
        unique_indices = list(dict.fromkeys(int(index) for index in indices))
        if len(unique_indices) != max_windows:
            raise EvaluationError("Even walk-forward window selection produced duplicate windows")
        return [windows[index] for index in unique_indices]

    raise EvaluationError(
        "window_selection must be one of: " + ", ".join(SUPPORTED_WINDOW_SELECTIONS)
    )


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


def _evaluate_walk_forward_window(
    *,
    window: EvaluationWindow,
    manifest: dict[str, Any],
    predictor: Any,
    evaluation_created_at: str,
    model_name: str,
    top_p: float,
    sample_count: int,
    window_selection: str,
    input_transform: str,
    lookback: int,
    pred_len: int,
    window_number: int,
    sma_window: int,
) -> dict[str, Any]:
    input_window = window.input_window.copy()
    x_df, rescale_refs = _transform_walk_forward_input(
        input_window,
        input_transform=input_transform,
    )
    x_timestamp = input_window["timestamp"]
    y_timestamp = pd.Series([pd.Timestamp(window.target_timestamp)])

    target_timestamp = pd.Timestamp(y_timestamp.iloc[0])
    input_end = pd.Timestamp(x_timestamp.iloc[-1])
    expected_target = input_end + timeframe_delta(window.timeframe)
    if input_end >= target_timestamp:
        raise EvaluationError("Walk-forward input window includes or overlaps its target")
    if target_timestamp != expected_target:
        raise EvaluationError(
            f"{window.timeframe} target timestamp does not follow input window end"
        )

    prediction = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    if len(prediction) != 1:
        raise EvaluationError("Kronos predictor must return exactly one row for pred_len=1")

    prediction_row = _rescale_prediction_row(
        prediction.iloc[0],
        input_transform=input_transform,
        refs=rescale_refs,
    )
    kronos_close = float(prediction_row["close"])
    current_close = float(window.current_close)
    target_close = float(window.target_close)
    metrics = _close_metric_values(
        current_close=current_close,
        target_close=target_close,
        kronos_close=kronos_close,
    )
    sma_close = float(input_window["close"].tail(sma_window).mean())
    metrics.update(
        _sma_metric_values(
            current_close=current_close,
            actual_return=metrics["actual_return"],
            target_close=target_close,
            sma_close=sma_close,
        )
    )

    return {
        "run_id": manifest["run_id"],
        "evaluation_created_at": evaluation_created_at,
        "exchange": manifest["exchange"],
        "symbol": manifest["symbol"],
        "timeframe": window.timeframe,
        "model_name": model_name,
        "top_p": top_p,
        "sample_count": sample_count,
        "window_selection": window_selection,
        "input_transform": input_transform,
        "lookback": lookback,
        "pred_len": pred_len,
        "window_number": window_number,
        "input_start_timestamp": window.input_start_timestamp,
        "input_end_timestamp": window.input_end_timestamp,
        "forecast_timestamp": window.target_timestamp,
        **metrics,
    }


def _transform_walk_forward_input(
    input_window: pd.DataFrame,
    *,
    input_transform: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    x_df = input_window[CLEAN_COLUMNS[1:]].copy()
    if input_transform == "raw":
        return x_df, {}

    if input_transform != "relative":
        raise EvaluationError(
            "input_transform must be one of: " + ", ".join(SUPPORTED_INPUT_TRANSFORMS)
        )

    reference_close = float(input_window["close"].iloc[-1])
    reference_volume = float(input_window["volume"].median())
    reference_amount = float(input_window["amount"].median())
    if reference_close == 0:
        raise EvaluationError("Relative input transform requires non-zero reference close")
    if reference_volume == 0:
        raise EvaluationError("Relative input transform requires non-zero median volume")
    if reference_amount == 0:
        raise EvaluationError("Relative input transform requires non-zero median amount")

    transformed = x_df.copy()
    for column in ("open", "high", "low", "close"):
        transformed[column] = transformed[column] / reference_close
    transformed["volume"] = transformed["volume"] / reference_volume
    transformed["amount"] = transformed["amount"] / reference_amount
    return transformed, {
        "close": reference_close,
        "volume": reference_volume,
        "amount": reference_amount,
    }


def _rescale_prediction_row(
    prediction_row: pd.Series,
    *,
    input_transform: str,
    refs: dict[str, float],
) -> pd.Series:
    if input_transform == "raw":
        return prediction_row

    if input_transform != "relative":
        raise EvaluationError(
            "input_transform must be one of: " + ", ".join(SUPPORTED_INPUT_TRANSFORMS)
        )

    rescaled = prediction_row.copy()
    for column in ("open", "high", "low", "close"):
        rescaled[column] = float(rescaled[column]) * refs["close"]
    rescaled["volume"] = float(rescaled["volume"]) * refs["volume"]
    rescaled["amount"] = float(rescaled["amount"]) * refs["amount"]
    return rescaled


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


def _close_metric_values(
    *,
    current_close: float,
    target_close: float,
    kronos_close: float,
) -> dict[str, Any]:
    kronos_error = kronos_close - target_close
    naive_close = current_close
    naive_error = naive_close - target_close
    actual_return = _safe_return(target_close, current_close)
    forecasted_return = _safe_return(kronos_close, current_close)
    naive_return = 0.0
    return {
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


def _sma_metric_values(
    *,
    current_close: float,
    actual_return: float,
    target_close: float,
    sma_close: float,
) -> dict[str, Any]:
    sma_error = sma_close - target_close
    sma_return = _safe_return(sma_close, current_close)
    return {
        "sma_close": sma_close,
        "sma_close_error": sma_error,
        "sma_absolute_error": abs(sma_error),
        "sma_squared_error": sma_error**2,
        "sma_return": sma_return,
        "sma_direction_hit": bool(np.sign(sma_return) == np.sign(actual_return)),
    }


def _normalize_timestamp(timestamp: pd.Timestamp | None) -> pd.Timestamp:
    if timestamp is None:
        return pd.Timestamp.now(tz="UTC")
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return _normalize_timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def _sampling_slug(*, top_p: float, sample_count: int) -> str:
    top_p_slug = f"{top_p:g}".replace("-", "neg").replace(".", "p")
    return f"top-p-{top_p_slug}_sample-count-{sample_count}"


def _coerce_boolean_series(series: pd.Series, *, column: str) -> pd.Series:
    if series.dtype == bool:
        return series

    mapped = series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            1: True,
            0: False,
        }
    )
    if mapped.isna().any():
        raise EvaluationError(f"Walk-forward metrics contain invalid boolean values in {column}")
    return mapped.astype(bool)


def _summary_filename(metrics_path: Path) -> str:
    stem = metrics_path.stem
    if stem.endswith("_walk_forward_metrics"):
        stem = stem.removesuffix("_walk_forward_metrics") + "_walk_forward_summary"
    else:
        stem = stem + "_summary"
    return f"{stem}.csv"


def _diagnose_timeframe_group(
    *,
    model_name: str,
    top_p: float,
    sample_count: int,
    window_selection: str,
    input_transform: str,
    timeframe: str,
    group: pd.DataFrame,
    random_seed: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    kronos_mae = float(group["kronos_absolute_error"].mean())
    naive_mae = float(group["naive_absolute_error"].mean())
    sma_mae = float(group["sma_absolute_error"].mean())
    actual_direction = _direction_labels(group["actual_return"])
    predicted_direction = _direction_labels(group["forecasted_return"])
    random_direction = rng.choice(np.array(_DIRECTION_LABELS), size=len(group))

    row = {
        "model_name": model_name,
        "top_p": top_p,
        "sample_count": sample_count,
        "window_selection": window_selection,
        "input_transform": input_transform,
        "timeframe": timeframe,
        "rows": int(len(group)),
        "random_seed": random_seed,
        "kronos_mean_signed_error": float(group["kronos_close_error"].mean()),
        "naive_mean_signed_error": float(group["naive_close_error"].mean()),
        "sma_mean_signed_error": float(group["sma_close_error"].mean()),
        "kronos_median_absolute_error": float(group["kronos_absolute_error"].median()),
        "naive_median_absolute_error": float(group["naive_absolute_error"].median()),
        "sma_median_absolute_error": float(group["sma_absolute_error"].median()),
        "kronos_error_std": float(group["kronos_close_error"].std(ddof=0)),
        "naive_error_std": float(group["naive_close_error"].std(ddof=0)),
        "sma_error_std": float(group["sma_close_error"].std(ddof=0)),
        "kronos_vs_naive_mae_delta": kronos_mae - naive_mae,
        "kronos_vs_naive_mae_ratio": _safe_ratio(kronos_mae, naive_mae),
        "kronos_vs_sma_mae_delta": kronos_mae - sma_mae,
        "kronos_vs_sma_mae_ratio": _safe_ratio(kronos_mae, sma_mae),
        "average_actual_return": float(group["actual_return"].mean()),
        "average_forecasted_return": float(group["forecasted_return"].mean()),
        "random_directional_accuracy": float(np.mean(random_direction == actual_direction)),
    }
    row.update(_confusion_counts(actual_direction, predicted_direction))
    return row


def _direction_labels(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.where(numeric > 0, "up", np.where(numeric < 0, "down", "flat"))


def _confusion_counts(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {
        f"kronos_actual_{actual_label}_pred_{predicted_label}": int(
            np.sum((actual == actual_label) & (predicted == predicted_label))
        )
        for actual_label in _DIRECTION_LABELS
        for predicted_label in _DIRECTION_LABELS
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    if len(left_values) < 2:
        return float("nan")
    if left_values.std(ddof=0) == 0 or right_values.std(ddof=0) == 0:
        return float("nan")
    return float(left_values.corr(right_values))


def _target_formulation_columns(thresholds_bps: tuple[int, ...]) -> list[str]:
    return [
        *TARGET_FORMULATION_BASE_COLUMNS,
        *[
            column
            for threshold in thresholds_bps
            for column in (
                f"threshold_{threshold}_bps_rows",
                f"threshold_{threshold}_bps_kronos_directional_accuracy",
                f"threshold_{threshold}_bps_naive_directional_accuracy",
                f"threshold_{threshold}_bps_sma_directional_accuracy",
                f"kronos_beats_naive_threshold_{threshold}_bps_direction",
            )
        ],
        "kronos_beats_naive_any_threshold_direction",
    ]


def _analyze_target_formulation_group(
    *,
    model_name: str,
    top_p: float,
    sample_count: int,
    window_selection: str,
    input_transform: str,
    timeframe: str,
    group: pd.DataFrame,
    thresholds_bps: tuple[int, ...],
) -> dict[str, Any]:
    kronos_return_error = group["forecasted_return"] - group["actual_return"]
    naive_return_error = group["naive_return"] - group["actual_return"]
    sma_return_error = group["sma_return"] - group["actual_return"]

    kronos_return_mae = float(kronos_return_error.abs().mean())
    naive_return_mae = float(naive_return_error.abs().mean())
    kronos_close_mae = float(group["kronos_absolute_error"].mean())
    naive_close_mae = float(group["naive_absolute_error"].mean())

    row: dict[str, Any] = {
        "model_name": model_name,
        "top_p": top_p,
        "sample_count": sample_count,
        "window_selection": window_selection,
        "input_transform": input_transform,
        "timeframe": timeframe,
        "rows": int(len(group)),
        "kronos_close_mae": kronos_close_mae,
        "kronos_close_rmse": float(np.sqrt(group["kronos_squared_error"].mean())),
        "naive_close_mae": naive_close_mae,
        "naive_close_rmse": float(np.sqrt(group["naive_squared_error"].mean())),
        "sma_close_mae": float(group["sma_absolute_error"].mean()),
        "sma_close_rmse": float(np.sqrt(group["sma_squared_error"].mean())),
        "kronos_return_mae": kronos_return_mae,
        "kronos_return_rmse": _rmse(kronos_return_error),
        "naive_return_mae": naive_return_mae,
        "naive_return_rmse": _rmse(naive_return_error),
        "sma_return_mae": float(sma_return_error.abs().mean()),
        "sma_return_rmse": _rmse(sma_return_error),
        "kronos_mean_signed_close_error": float(group["kronos_close_error"].mean()),
        "kronos_median_signed_close_error": float(group["kronos_close_error"].median()),
        "naive_mean_signed_close_error": float(group["naive_close_error"].mean()),
        "naive_median_signed_close_error": float(group["naive_close_error"].median()),
        "sma_mean_signed_close_error": float(group["sma_close_error"].mean()),
        "sma_median_signed_close_error": float(group["sma_close_error"].median()),
        "kronos_mean_signed_return_error": float(kronos_return_error.mean()),
        "kronos_median_signed_return_error": float(kronos_return_error.median()),
        "naive_mean_signed_return_error": float(naive_return_error.mean()),
        "naive_median_signed_return_error": float(naive_return_error.median()),
        "sma_mean_signed_return_error": float(sma_return_error.mean()),
        "sma_median_signed_return_error": float(sma_return_error.median()),
        "forecast_actual_return_correlation": _safe_correlation(
            group["forecasted_return"],
            group["actual_return"],
        ),
        "kronos_beats_naive_close_mae": kronos_close_mae < naive_close_mae,
        "kronos_beats_naive_return_mae": kronos_return_mae < naive_return_mae,
    }

    threshold_beats: list[bool] = []
    for threshold in thresholds_bps:
        threshold_result = _threshold_direction_metrics(group, threshold_bps=threshold)
        row.update(threshold_result)
        threshold_beats.append(
            bool(threshold_result[f"kronos_beats_naive_threshold_{threshold}_bps_direction"])
        )
    row["kronos_beats_naive_any_threshold_direction"] = any(threshold_beats)
    return row


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _threshold_direction_metrics(group: pd.DataFrame, *, threshold_bps: int) -> dict[str, Any]:
    threshold_return = threshold_bps / 10_000
    threshold_rows = group.loc[group["actual_return"].abs() >= threshold_return]
    prefix = f"threshold_{threshold_bps}_bps"
    if threshold_rows.empty:
        kronos_accuracy = float("nan")
        naive_accuracy = float("nan")
        sma_accuracy = float("nan")
        beats_naive = False
    else:
        actual_direction = np.sign(threshold_rows["actual_return"].to_numpy(dtype=float))
        kronos_accuracy = _directional_accuracy(
            actual_direction,
            threshold_rows["forecasted_return"],
        )
        naive_accuracy = _directional_accuracy(actual_direction, threshold_rows["naive_return"])
        sma_accuracy = _directional_accuracy(actual_direction, threshold_rows["sma_return"])
        beats_naive = kronos_accuracy > naive_accuracy

    return {
        f"{prefix}_rows": int(len(threshold_rows)),
        f"{prefix}_kronos_directional_accuracy": kronos_accuracy,
        f"{prefix}_naive_directional_accuracy": naive_accuracy,
        f"{prefix}_sma_directional_accuracy": sma_accuracy,
        f"kronos_beats_naive_threshold_{threshold_bps}_bps_direction": beats_naive,
    }


def _directional_accuracy(actual_direction: np.ndarray, predicted_returns: pd.Series) -> float:
    predicted_direction = np.sign(pd.to_numeric(predicted_returns, errors="coerce").to_numpy(dtype=float))
    return float(np.mean(predicted_direction == actual_direction))


def _forecast_calibration_columns(thresholds_bps: tuple[int, ...]) -> list[str]:
    return [
        *FORECAST_CALIBRATION_BASE_COLUMNS,
        *[
            column
            for threshold in thresholds_bps
            for column in (
                f"threshold_{threshold}_bps_rows",
                f"threshold_{threshold}_bps_uncalibrated_directional_accuracy",
                f"threshold_{threshold}_bps_bias_directional_accuracy",
                f"threshold_{threshold}_bps_linear_directional_accuracy",
                f"threshold_{threshold}_bps_naive_directional_accuracy",
                f"threshold_{threshold}_bps_sma_directional_accuracy",
                f"uncalibrated_beats_naive_threshold_{threshold}_bps_direction",
                f"bias_beats_naive_threshold_{threshold}_bps_direction",
                f"linear_beats_naive_threshold_{threshold}_bps_direction",
            )
        ],
        "uncalibrated_beats_naive_any_threshold_direction",
        "bias_beats_naive_any_threshold_direction",
        "linear_beats_naive_any_threshold_direction",
    ]


def _analyze_forecast_calibration_group(
    *,
    model_name: str,
    top_p: float,
    sample_count: int,
    window_selection: str,
    input_transform: str,
    timeframe: str,
    group: pd.DataFrame,
    train_fraction: float,
    thresholds_bps: tuple[int, ...],
) -> dict[str, Any]:
    ordered = group.sort_values("forecast_timestamp", kind="mergesort").reset_index(drop=True)
    split_index = int(np.floor(len(ordered) * train_fraction))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    if len(train) < 2 or len(test) < 2:
        raise EvaluationError(
            f"Forecast calibration requires at least 2 train and 2 test rows for {model_name} {timeframe}"
        )

    bias_correction = float((train["actual_return"] - train["forecasted_return"]).mean())
    linear_alpha, linear_beta, linear_degenerate = _fit_linear_return_calibration(train)

    predictions = {
        "uncalibrated": test["forecasted_return"],
        "bias": test["forecasted_return"] + bias_correction,
        "linear": linear_alpha + linear_beta * test["forecasted_return"],
        "naive": test["naive_return"],
        "sma": test["sma_return"],
    }
    method_metrics = {
        method: _score_return_prediction(
            predicted_return=predicted_return,
            actual_return=test["actual_return"],
            current_close=test["current_close"],
        )
        for method, predicted_return in predictions.items()
    }

    row: dict[str, Any] = {
        "model_name": model_name,
        "top_p": top_p,
        "sample_count": sample_count,
        "window_selection": window_selection,
        "input_transform": input_transform,
        "timeframe": timeframe,
        "train_fraction": train_fraction,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start_timestamp": _format_timestamp(train["forecast_timestamp"].iloc[0]),
        "train_end_timestamp": _format_timestamp(train["forecast_timestamp"].iloc[-1]),
        "test_start_timestamp": _format_timestamp(test["forecast_timestamp"].iloc[0]),
        "test_end_timestamp": _format_timestamp(test["forecast_timestamp"].iloc[-1]),
        "bias_correction": bias_correction,
        "linear_alpha": linear_alpha,
        "linear_beta": linear_beta,
        "linear_degenerate": linear_degenerate,
        "test_forecast_actual_return_correlation": _safe_correlation(
            test["forecasted_return"],
            test["actual_return"],
        ),
    }
    for method, metrics in method_metrics.items():
        row.update(
            {
                f"{method}_return_mae": metrics["return_mae"],
                f"{method}_return_rmse": metrics["return_rmse"],
                f"{method}_close_mae": metrics["close_mae"],
                f"{method}_close_rmse": metrics["close_rmse"],
            }
        )

    for method in ("uncalibrated", "bias", "linear"):
        row[f"{method}_beats_naive_return_mae"] = (
            row[f"{method}_return_mae"] < row["naive_return_mae"]
        )
        row[f"{method}_beats_naive_close_mae"] = row[f"{method}_close_mae"] < row["naive_close_mae"]

    threshold_beats = {"uncalibrated": [], "bias": [], "linear": []}
    for threshold in thresholds_bps:
        threshold_result = _calibration_threshold_direction_metrics(
            test,
            predictions=predictions,
            threshold_bps=threshold,
        )
        row.update(threshold_result)
        for method in threshold_beats:
            threshold_beats[method].append(
                bool(threshold_result[f"{method}_beats_naive_threshold_{threshold}_bps_direction"])
            )
    for method, beats in threshold_beats.items():
        row[f"{method}_beats_naive_any_threshold_direction"] = any(beats)
    return row


def _fit_linear_return_calibration(train: pd.DataFrame) -> tuple[float, float, bool]:
    forecast = train["forecasted_return"].to_numpy(dtype=float)
    actual = train["actual_return"].to_numpy(dtype=float)
    forecast_mean = float(forecast.mean())
    actual_mean = float(actual.mean())
    centered_forecast = forecast - forecast_mean
    denominator = float(np.sum(centered_forecast**2))
    if denominator == 0:
        return actual_mean, 0.0, True
    beta = float(np.sum(centered_forecast * (actual - actual_mean)) / denominator)
    alpha = float(actual_mean - beta * forecast_mean)
    return alpha, beta, False


def _score_return_prediction(
    *,
    predicted_return: pd.Series,
    actual_return: pd.Series,
    current_close: pd.Series,
) -> dict[str, float]:
    return_error = predicted_return - actual_return
    close_error = return_error * current_close
    return {
        "return_mae": float(return_error.abs().mean()),
        "return_rmse": _rmse(return_error),
        "close_mae": float(close_error.abs().mean()),
        "close_rmse": _rmse(close_error),
    }


def _calibration_threshold_direction_metrics(
    test: pd.DataFrame,
    *,
    predictions: dict[str, pd.Series],
    threshold_bps: int,
) -> dict[str, Any]:
    threshold_return = threshold_bps / 10_000
    threshold_rows = test.loc[test["actual_return"].abs() >= threshold_return]
    prefix = f"threshold_{threshold_bps}_bps"
    result: dict[str, Any] = {f"{prefix}_rows": int(len(threshold_rows))}
    if threshold_rows.empty:
        for method in ("uncalibrated", "bias", "linear", "naive", "sma"):
            result[f"{prefix}_{method}_directional_accuracy"] = float("nan")
        for method in ("uncalibrated", "bias", "linear"):
            result[f"{method}_beats_naive_threshold_{threshold_bps}_bps_direction"] = False
        return result

    actual_direction = np.sign(threshold_rows["actual_return"].to_numpy(dtype=float))
    for method in ("uncalibrated", "bias", "linear", "naive", "sma"):
        result[f"{prefix}_{method}_directional_accuracy"] = _directional_accuracy(
            actual_direction,
            predictions[method].loc[threshold_rows.index],
        )
    for method in ("uncalibrated", "bias", "linear"):
        result[f"{method}_beats_naive_threshold_{threshold_bps}_bps_direction"] = (
            result[f"{prefix}_{method}_directional_accuracy"]
            > result[f"{prefix}_naive_directional_accuracy"]
        )
    return result


def _diagnostic_filename(metrics_path: Path) -> str:
    stem = metrics_path.stem
    if stem.endswith("_walk_forward_metrics"):
        stem = stem.removesuffix("_walk_forward_metrics") + "_walk_forward_diagnostics"
    else:
        stem = stem + "_diagnostics"
    return f"{stem}.csv"
