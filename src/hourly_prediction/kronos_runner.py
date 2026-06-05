from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.kronos_runtime import (
    DEFAULT_KRONOS_MODEL,
    DEFAULT_KRONOS_TOKENIZER,
    DEFAULT_MAX_CONTEXT,
    KronosRuntimeError,
    _check_required_classes,
    _import_kronos_model,
)
from hourly_prediction.validation import timeframe_delta


DEFAULT_LOOKBACK = 512
DEFAULT_PRED_LEN = 1
DEFAULT_DEVICE = "cuda:0"
FORECAST_COLUMNS = [
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

PredictorLoader = Callable[..., Any]


class KronosForecastError(ValueError):
    """Raised when a manifest or clean dataset cannot produce a valid forecast."""


@dataclass(frozen=True)
class ForecastRun:
    manifest_path: Path
    output_path: Path
    rows: int


def resolve_manifest_path(manifest: str | Path, *, manifest_dir: Path) -> Path:
    if str(manifest) == "latest":
        candidates = sorted(
            manifest_dir.glob("*_manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise KronosForecastError(f"No manifest files found under: {manifest_dir}")
        return candidates[0]

    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise KronosForecastError(f"Manifest does not exist: {manifest_path}")
    return manifest_path


def run_kronos_forecast(
    *,
    manifest: str | Path,
    manifest_dir: Path,
    output_dir: Path,
    kronos_repo_path: Path,
    device: str = DEFAULT_DEVICE,
    lookback: int = DEFAULT_LOOKBACK,
    pred_len: int = DEFAULT_PRED_LEN,
    model_name: str = DEFAULT_KRONOS_MODEL,
    tokenizer_name: str = DEFAULT_KRONOS_TOKENIZER,
    now: pd.Timestamp | None = None,
    predictor_loader: PredictorLoader = None,
) -> ForecastRun:
    if pred_len != 1:
        raise KronosForecastError("Only pred_len=1 is supported in Phase 3A")
    if lookback <= 0:
        raise KronosForecastError("lookback must be positive")

    manifest_path = resolve_manifest_path(manifest, manifest_dir=manifest_dir)
    payload = _load_manifest(manifest_path)
    forecast_created_at = _format_timestamp(_normalize_timestamp(now))
    predictor = (predictor_loader or load_kronos_predictor)(
        kronos_repo_path=kronos_repo_path,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
        max_context=lookback,
    )

    rows: list[dict[str, Any]] = []
    for dataset in payload["datasets"]:
        rows.append(
            _forecast_dataset(
                dataset=dataset,
                manifest=payload,
                predictor=predictor,
                forecast_created_at=forecast_created_at,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                device=device,
                lookback=lookback,
                pred_len=pred_len,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['run_id']}_{_model_slug(model_name)}_forecast.csv"
    pd.DataFrame(rows, columns=FORECAST_COLUMNS).to_csv(output_path, index=False)
    return ForecastRun(manifest_path=manifest_path, output_path=output_path, rows=len(rows))


def load_kronos_predictor(
    *,
    kronos_repo_path: Path,
    model_name: str,
    tokenizer_name: str,
    device: str,
    max_context: int = DEFAULT_MAX_CONTEXT,
) -> Any:
    try:
        model_module = _import_kronos_model(kronos_repo_path)
        _check_required_classes(model_module)
        tokenizer = model_module.KronosTokenizer.from_pretrained(tokenizer_name)
        model = model_module.Kronos.from_pretrained(model_name)
        return model_module.KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=max_context,
        )
    except KronosRuntimeError:
        raise
    except Exception as exc:
        raise KronosForecastError(f"Failed to load Kronos predictor: {exc}") from exc


def _forecast_dataset(
    *,
    dataset: dict[str, Any],
    manifest: dict[str, Any],
    predictor: Any,
    forecast_created_at: str,
    model_name: str,
    tokenizer_name: str,
    device: str,
    lookback: int,
    pred_len: int,
) -> dict[str, Any]:
    timeframe = dataset["timeframe"]
    clean = _load_clean_candles(Path(dataset["clean_path"]), timeframe=timeframe)
    if len(clean) < lookback:
        raise KronosForecastError(
            f"{timeframe} forecast requires at least {lookback} clean rows; found {len(clean)}"
        )

    window = clean.tail(lookback).reset_index(drop=True)
    x_df = window[CLEAN_COLUMNS[1:]].copy()
    x_timestamp = window["timestamp"]
    forecast_timestamp = x_timestamp.iloc[-1] + timeframe_delta(timeframe)
    y_timestamp = pd.Series([forecast_timestamp])
    prediction = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    prediction_row = prediction.iloc[0]

    return {
        "run_id": manifest["run_id"],
        "forecast_created_at": forecast_created_at,
        "exchange": manifest["exchange"],
        "symbol": manifest["symbol"],
        "timeframe": timeframe,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "device": device,
        "lookback": lookback,
        "pred_len": pred_len,
        "input_start_timestamp": _format_timestamp(x_timestamp.iloc[0]),
        "input_end_timestamp": _format_timestamp(x_timestamp.iloc[-1]),
        "forecast_timestamp": _format_timestamp(forecast_timestamp),
        "open": float(prediction_row["open"]),
        "high": float(prediction_row["high"]),
        "low": float(prediction_row["low"]),
        "close": float(prediction_row["close"]),
        "volume": float(prediction_row["volume"]),
        "amount": float(prediction_row["amount"]),
    }


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"run_id", "exchange", "symbol", "datasets"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KronosForecastError(f"Manifest missing required field(s): {', '.join(missing)}")
    if not isinstance(payload["datasets"], list) or not payload["datasets"]:
        raise KronosForecastError("Manifest must contain at least one dataset")
    return payload


def _load_clean_candles(clean_path: Path, *, timeframe: str) -> pd.DataFrame:
    if not clean_path.exists():
        raise KronosForecastError(f"Clean candle file does not exist: {clean_path}")

    clean = pd.read_csv(clean_path)
    if list(clean.columns) != CLEAN_COLUMNS:
        raise KronosForecastError(
            "Clean candle schema mismatch; expected: " + ", ".join(CLEAN_COLUMNS)
        )

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    if clean["timestamp"].isna().any():
        raise KronosForecastError(f"{timeframe} clean file contains invalid timestamps")

    for column in CLEAN_COLUMNS[1:]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[CLEAN_COLUMNS[1:]].isna().any().any():
        raise KronosForecastError(f"{timeframe} clean file contains invalid numeric values")

    return clean


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
