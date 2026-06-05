from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from hourly_prediction.config import RAW_COLUMNS


def fetch_ohlcv(
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch public OHLCV candles from an exchange through CCXT."""

    import ccxt

    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise ValueError(f"Unsupported CCXT exchange: {exchange_id}") from exc

    exchange = exchange_class({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    fetched_at = pd.Timestamp.now(tz="UTC")
    fetched_at_text = _format_timestamp(fetched_at)

    raw = pd.DataFrame(
        rows,
        columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    raw["exchange"] = exchange_id
    raw["symbol"] = symbol
    raw["timeframe"] = timeframe
    raw["fetched_at"] = fetched_at_text

    return raw[RAW_COLUMNS]


def raw_output_path(
    *,
    output_dir: Path,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    fetched_at: pd.Timestamp | None = None,
) -> Path:
    fetched_at = fetched_at or pd.Timestamp.now(tz="UTC")
    filename = (
        f"{_safe_name(exchange_id)}_"
        f"{_safe_name(symbol)}_"
        f"{_safe_name(timeframe)}_"
        f"{fetched_at.strftime('%Y%m%dT%H%M%SZ')}_raw.csv"
    )
    return output_dir / filename


def write_raw_candles(raw: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    return output_path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
