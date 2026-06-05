from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hourly_prediction.config import DEFAULT_TIMEFRAMES
from hourly_prediction.data import fetch_ohlcv, raw_output_path, write_raw_candles
from hourly_prediction.validation import validate_candles, write_clean_candles

FetchCandles = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class RefreshResult:
    timeframe: str
    raw_path: Path
    clean_path: Path
    clean_rows: int


def refresh_candles(
    *,
    exchange_id: str,
    symbol: str,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    limit: int,
    raw_dir: Path,
    clean_dir: Path,
    fetched_at: pd.Timestamp | None = None,
    now: pd.Timestamp | None = None,
    fetcher: FetchCandles = fetch_ohlcv,
) -> list[RefreshResult]:
    """Fetch and validate each timeframe, returning written artifact paths."""

    run_timestamp = _normalize_timestamp(fetched_at)
    results: list[RefreshResult] = []

    for timeframe in timeframes:
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

        results.append(
            RefreshResult(
                timeframe=timeframe,
                raw_path=raw_path,
                clean_path=clean_path,
                clean_rows=len(clean),
            )
        )

    return results


def _normalize_timestamp(timestamp: pd.Timestamp | None) -> pd.Timestamp:
    if timestamp is None:
        return pd.Timestamp.now(tz="UTC")

    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")
