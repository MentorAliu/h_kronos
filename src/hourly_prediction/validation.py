from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS, SUPPORTED_TIMEFRAMES


class CandleValidationError(ValueError):
    """Raised when an OHLCV candle dataset violates the clean data contract."""


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(seconds=SUPPORTED_TIMEFRAMES[timeframe])
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        raise CandleValidationError(
            f"Unsupported timeframe {timeframe!r}; expected one of: {supported}"
        ) from exc


def validate_candles(
    df: pd.DataFrame,
    *,
    timeframe: str,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return model-ready candles or raise when validation fails."""

    delta = timeframe_delta(timeframe)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise CandleValidationError(f"Missing required columns: {', '.join(missing)}")

    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    if clean["timestamp"].isna().any():
        raise CandleValidationError("Invalid timestamp value found")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    if "amount" in clean.columns:
        numeric_columns.append("amount")

    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    numeric_values = clean[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise CandleValidationError("Invalid numeric value found")

    if "amount" not in clean.columns:
        clean["amount"] = clean["close"] * clean["volume"]

    clean = clean[CLEAN_COLUMNS].copy()

    if clean["timestamp"].duplicated().any():
        raise CandleValidationError("Found duplicate timestamps")

    if not clean["timestamp"].is_monotonic_increasing:
        raise CandleValidationError("Timestamps must be sorted ascending")

    if len(clean) > 1:
        intervals = clean["timestamp"].diff().dropna()
        if not (intervals == delta).all():
            raise CandleValidationError(f"Found gap or irregular {timeframe} interval")

    reference_now = _normalize_now(now)
    closed_mask = clean["timestamp"] + delta <= reference_now
    clean = clean.loc[closed_mask].reset_index(drop=True)

    return clean


def write_clean_candles(clean: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = clean.copy()
    serializable["timestamp"] = serializable["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    serializable.to_csv(output_path, index=False)
    return output_path


def _normalize_now(now: pd.Timestamp | None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz="UTC")

    timestamp = pd.Timestamp(now)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
