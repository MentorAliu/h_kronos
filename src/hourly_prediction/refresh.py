from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS, DEFAULT_TIMEFRAMES
from hourly_prediction.data import fetch_ohlcv, raw_output_path, write_raw_candles
from hourly_prediction.validation import validate_candles, write_clean_candles

FetchCandles = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class RefreshResult:
    timeframe: str
    raw_path: Path
    clean_path: Path
    clean_rows: int
    start_timestamp: str | None
    end_timestamp: str | None


@dataclass(frozen=True)
class RefreshRun:
    run_id: str
    manifest_path: Path
    results: list[RefreshResult]


def refresh_candles(
    *,
    exchange_id: str,
    symbol: str,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    limit: int,
    raw_dir: Path,
    clean_dir: Path,
    manifest_dir: Path = Path("outputs/manifests"),
    fetched_at: pd.Timestamp | None = None,
    now: pd.Timestamp | None = None,
    fetcher: FetchCandles = fetch_ohlcv,
) -> RefreshRun:
    """Fetch and validate each timeframe, returning the written run artifacts."""

    run_timestamp = _normalize_timestamp(fetched_at)
    run_id = _run_id(
        exchange_id=exchange_id,
        symbol=symbol,
        timestamp=run_timestamp,
    )
    manifest_path = manifest_dir / f"{run_id}_manifest.json"
    requested_timeframes = tuple(timeframes)
    results: list[RefreshResult] = []

    for timeframe in requested_timeframes:
        raw = fetcher(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        raw_path = raw_output_path(
            output_dir=raw_dir,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            fetched_at=run_timestamp,
        )
        write_raw_candles(raw, raw_path)

        clean = validate_candles(raw, timeframe=timeframe, now=now)
        clean_path = clean_dir / f"{raw_path.stem.removesuffix('_raw')}_clean.csv"
        write_clean_candles(clean, clean_path)
        start_timestamp, end_timestamp = _clean_timestamp_bounds(clean)

        results.append(
            RefreshResult(
                timeframe=timeframe,
                raw_path=raw_path,
                clean_path=clean_path,
                clean_rows=len(clean),
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        )

    _write_manifest(
        manifest_path=manifest_path,
        run_id=run_id,
        created_at=_format_timestamp(run_timestamp),
        exchange_id=exchange_id,
        symbol=symbol,
        limit=limit,
        timeframes=requested_timeframes,
        results=results,
    )
    return RefreshRun(run_id=run_id, manifest_path=manifest_path, results=results)


def _write_manifest(
    *,
    manifest_path: Path,
    run_id: str,
    created_at: str,
    exchange_id: str,
    symbol: str,
    limit: int,
    timeframes: tuple[str, ...],
    results: list[RefreshResult],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at": created_at,
        "exchange": exchange_id,
        "symbol": symbol,
        "limit": limit,
        "timeframes": list(timeframes),
        "datasets": [
            {
                "timeframe": result.timeframe,
                "raw_path": str(result.raw_path),
                "clean_path": str(result.clean_path),
                "clean_rows": result.clean_rows,
                "start_timestamp": result.start_timestamp,
                "end_timestamp": result.end_timestamp,
                "schema": list(CLEAN_COLUMNS),
                "valid": True,
            }
            for result in results
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _clean_timestamp_bounds(clean: pd.DataFrame) -> tuple[str | None, str | None]:
    if clean.empty:
        return None, None

    start = _format_timestamp(pd.Timestamp(clean["timestamp"].iloc[0]))
    end = _format_timestamp(pd.Timestamp(clean["timestamp"].iloc[-1]))
    return start, end


def _normalize_timestamp(timestamp: pd.Timestamp | None) -> pd.Timestamp:
    if timestamp is None:
        return pd.Timestamp.now(tz="UTC")

    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return _normalize_timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(*, exchange_id: str, symbol: str, timestamp: pd.Timestamp) -> str:
    return (
        f"{_safe_name(exchange_id)}_"
        f"{_safe_name(symbol)}_"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)
