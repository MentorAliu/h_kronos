from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS, RAW_COLUMNS, SUPPORTED_TIMEFRAMES
from hourly_prediction.data import write_raw_candles
from hourly_prediction.validation import validate_candles, write_clean_candles

BINANCE_PUBLIC_BASE_URL = "https://data.binance.vision"
BINANCE_PUBLIC_EXCHANGE_ID = "binance_public"
BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

DownloadBytes = Callable[[str], bytes]


class BinanceBackfillError(ValueError):
    """Raised when Binance public kline data cannot be parsed safely."""


@dataclass(frozen=True)
class BinanceBackfillResult:
    timeframe: str
    raw_path: Path
    clean_path: Path
    clean_rows: int
    start_timestamp: str | None
    end_timestamp: str | None
    months: tuple[str, ...]


@dataclass(frozen=True)
class BinanceBackfillRun:
    run_id: str
    manifest_path: Path
    results: list[BinanceBackfillResult]


def binance_monthly_klines_url(
    *,
    symbol: str,
    timeframe: str,
    month: str,
    base_url: str = BINANCE_PUBLIC_BASE_URL,
) -> str:
    symbol = _normalize_binance_symbol(symbol)
    _validate_timeframe(timeframe)
    _normalize_month(month)
    return (
        f"{base_url.rstrip('/')}/data/spot/monthly/klines/"
        f"{symbol}/{timeframe}/{symbol}-{timeframe}-{month}.zip"
    )


def download_url(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed public data URL.
        return response.read()


def parse_binance_kline_zip(
    payload: bytes,
    *,
    exchange_id: str,
    project_symbol: str,
    timeframe: str,
    fetched_at: pd.Timestamp,
) -> pd.DataFrame:
    _validate_timeframe(timeframe)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise BinanceBackfillError(
                f"Expected exactly one CSV in Binance kline ZIP; found {len(csv_names)}"
            )
        with archive.open(csv_names[0]) as handle:
            rows = pd.read_csv(handle, header=None)

    if rows.empty:
        raise BinanceBackfillError("Binance kline ZIP contains no rows")

    rows = _drop_optional_header(rows)
    if rows.shape[1] < len(BINANCE_KLINE_COLUMNS):
        raise BinanceBackfillError(
            f"Binance kline CSV has {rows.shape[1]} columns; expected at least {len(BINANCE_KLINE_COLUMNS)}"
        )

    rows = rows.iloc[:, : len(BINANCE_KLINE_COLUMNS)].copy()
    rows.columns = BINANCE_KLINE_COLUMNS
    open_time = pd.to_numeric(rows["open_time"], errors="coerce")
    if open_time.isna().any():
        raise BinanceBackfillError("Binance kline CSV contains invalid open_time values")

    timestamp_unit = _infer_timestamp_unit(open_time)
    timestamps = pd.to_datetime(open_time, unit=timestamp_unit, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise BinanceBackfillError("Binance kline CSV contains unparseable timestamps")

    raw = pd.DataFrame(
        {
            "timestamp": timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": rows["open"],
            "high": rows["high"],
            "low": rows["low"],
            "close": rows["close"],
            "volume": rows["volume"],
            "exchange": exchange_id,
            "symbol": project_symbol,
            "timeframe": timeframe,
            "fetched_at": _format_timestamp(fetched_at),
        }
    )
    return raw[RAW_COLUMNS]


def backfill_binance_klines(
    *,
    symbol: str,
    timeframes: Iterable[str],
    start_month: str,
    end_month: str,
    raw_dir: Path,
    clean_dir: Path,
    manifest_dir: Path,
    fetched_at: pd.Timestamp | None = None,
    now: pd.Timestamp | None = None,
    downloader: DownloadBytes = download_url,
) -> BinanceBackfillRun:
    source_symbol = _normalize_binance_symbol(symbol)
    project_symbol = _project_symbol(source_symbol)
    requested_timeframes = tuple(timeframes)
    if not requested_timeframes:
        raise BinanceBackfillError("At least one timeframe is required")
    for timeframe in requested_timeframes:
        _validate_timeframe(timeframe)

    months = _month_range(start_month, end_month)
    run_timestamp = _normalize_timestamp(fetched_at)
    run_id = _run_id(symbol=source_symbol, timestamp=run_timestamp)
    manifest_path = manifest_dir / f"{run_id}_manifest.json"
    results: list[BinanceBackfillResult] = []

    for timeframe in requested_timeframes:
        monthly_raw = []
        for month in months:
            url = binance_monthly_klines_url(
                symbol=source_symbol,
                timeframe=timeframe,
                month=month,
            )
            monthly_raw.append(
                parse_binance_kline_zip(
                    downloader(url),
                    exchange_id=BINANCE_PUBLIC_EXCHANGE_ID,
                    project_symbol=project_symbol,
                    timeframe=timeframe,
                    fetched_at=run_timestamp,
                )
            )

        raw = pd.concat(monthly_raw, ignore_index=True)
        raw_path = _raw_output_path(
            output_dir=raw_dir,
            run_timestamp=run_timestamp,
            symbol=source_symbol,
            timeframe=timeframe,
            start_month=months[0],
            end_month=months[-1],
        )
        write_raw_candles(raw, raw_path)

        clean = validate_candles(raw, timeframe=timeframe, now=now)
        clean_path = clean_dir / f"{raw_path.stem.removesuffix('_raw')}_clean.csv"
        write_clean_candles(clean, clean_path)
        start_timestamp, end_timestamp = _clean_timestamp_bounds(clean)
        results.append(
            BinanceBackfillResult(
                timeframe=timeframe,
                raw_path=raw_path,
                clean_path=clean_path,
                clean_rows=len(clean),
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                months=months,
            )
        )

    _write_manifest(
        manifest_path=manifest_path,
        run_id=run_id,
        created_at=_format_timestamp(run_timestamp),
        source_symbol=source_symbol,
        project_symbol=project_symbol,
        start_month=months[0],
        end_month=months[-1],
        timeframes=requested_timeframes,
        results=results,
    )
    return BinanceBackfillRun(run_id=run_id, manifest_path=manifest_path, results=results)


def _write_manifest(
    *,
    manifest_path: Path,
    run_id: str,
    created_at: str,
    source_symbol: str,
    project_symbol: str,
    start_month: str,
    end_month: str,
    timeframes: tuple[str, ...],
    results: list[BinanceBackfillResult],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "exchange": BINANCE_PUBLIC_EXCHANGE_ID,
        "symbol": project_symbol,
        "source_symbol": source_symbol,
        "start_month": start_month,
        "end_month": end_month,
        "timeframes": list(timeframes),
        "datasets": [
            {
                "timeframe": result.timeframe,
                "raw_path": str(result.raw_path),
                "clean_path": str(result.clean_path),
                "clean_rows": result.clean_rows,
                "start_timestamp": result.start_timestamp,
                "end_timestamp": result.end_timestamp,
                "months": list(result.months),
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


def _drop_optional_header(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    first_value = str(rows.iloc[0, 0]).strip().lower()
    if first_value in {"open_time", "open time"}:
        return rows.iloc[1:].reset_index(drop=True)
    return rows


def _infer_timestamp_unit(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise BinanceBackfillError("Binance kline CSV contains invalid timestamp values")

    abs_values = numeric.abs()
    microseconds = abs_values >= 10**15
    milliseconds = (abs_values >= 10**12) & (abs_values < 10**15)
    if microseconds.all():
        return "us"
    if milliseconds.all():
        return "ms"
    if (microseconds | milliseconds).all():
        raise BinanceBackfillError("Binance kline CSV contains mixed timestamp units")
    raise BinanceBackfillError("Binance kline CSV contains unsupported timestamp magnitudes")


def _month_range(start_month: str, end_month: str) -> tuple[str, ...]:
    start = _normalize_month(start_month)
    end = _normalize_month(end_month)
    if start > end:
        raise BinanceBackfillError("start_month must be before or equal to end_month")
    return tuple(
        period.strftime("%Y-%m")
        for period in pd.period_range(start=start, end=end, freq="M")
    )


def _normalize_month(month: str) -> pd.Period:
    try:
        return pd.Period(month, freq="M")
    except ValueError as exc:
        raise BinanceBackfillError(f"Invalid month {month!r}; expected YYYY-MM") from exc


def _validate_timeframe(timeframe: str) -> None:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        raise BinanceBackfillError(f"Unsupported timeframe {timeframe!r}; expected one of: {supported}")


def _project_symbol(source_symbol: str) -> str:
    if source_symbol.endswith("USDT"):
        return f"{source_symbol[:-4]}/USDT"
    return source_symbol


def _normalize_binance_symbol(symbol: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", symbol).upper()
    if not normalized:
        raise BinanceBackfillError("symbol must not be empty")
    return normalized


def _raw_output_path(
    *,
    output_dir: Path,
    run_timestamp: pd.Timestamp,
    symbol: str,
    timeframe: str,
    start_month: str,
    end_month: str,
) -> Path:
    filename = (
        f"binancepublic_{symbol}_{_safe_name(timeframe)}_"
        f"{start_month.replace('-', '')}_{end_month.replace('-', '')}_"
        f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}_raw.csv"
    )
    return output_dir / filename


def _clean_timestamp_bounds(clean: pd.DataFrame) -> tuple[str | None, str | None]:
    if clean.empty:
        return None, None

    start = _format_timestamp(pd.Timestamp(clean["timestamp"].iloc[0]))
    end = _format_timestamp(pd.Timestamp(clean["timestamp"].iloc[-1]))
    return start, end


def _run_id(*, symbol: str, timestamp: pd.Timestamp) -> str:
    return f"binancepublic_{symbol}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def _normalize_timestamp(timestamp: pd.Timestamp | None) -> pd.Timestamp:
    if timestamp is None:
        return pd.Timestamp.now(tz="UTC")
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return _normalize_timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)
