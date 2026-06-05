from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.refresh import refresh_candles
from hourly_prediction.validation import CandleValidationError


def stub_fetcher(
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    freq = "1h" if timeframe == "1h" else "15min"
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=limit, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": [100.0 + i for i in range(limit)],
            "high": [101.0 + i for i in range(limit)],
            "low": [99.0 + i for i in range(limit)],
            "close": [100.5 + i for i in range(limit)],
            "volume": [10.0 + i for i in range(limit)],
            "exchange": exchange_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "fetched_at": "2026-06-01T05:00:00Z",
        }
    )


def failing_fetcher(
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    raw = stub_fetcher(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )
    if timeframe == "15m":
        raw.loc[1, "timestamp"] = raw.loc[0, "timestamp"]
    return raw


def test_refresh_candles_fetches_validates_and_writes_both_timeframes(tmp_path) -> None:
    run = refresh_candles(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframes=("1h", "15m"),
        limit=3,
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        manifest_dir=tmp_path / "manifests",
        fetched_at=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
        now=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
        fetcher=stub_fetcher,
    )

    results = run.results
    assert [result.timeframe for result in results] == ["1h", "15m"]
    assert run.manifest_path == (
        tmp_path / "manifests" / "binance_BTCUSDT_20260601T050000Z_manifest.json"
    )

    for result in results:
        assert result.raw_path.exists()
        assert result.clean_path.exists()
        clean = pd.read_csv(result.clean_path)
        assert list(clean.columns) == CLEAN_COLUMNS
        assert result.clean_rows == 3

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "binance_BTCUSDT_20260601T050000Z"
    assert manifest["created_at"] == "2026-06-01T05:00:00Z"
    assert manifest["exchange"] == "binance"
    assert manifest["symbol"] == "BTC/USDT"
    assert manifest["limit"] == 3
    assert manifest["timeframes"] == ["1h", "15m"]

    assert [dataset["timeframe"] for dataset in manifest["datasets"]] == ["1h", "15m"]
    for dataset in manifest["datasets"]:
        assert Path(dataset["raw_path"]).exists()
        assert Path(dataset["clean_path"]).exists()
        assert dataset["clean_rows"] == 3
        assert dataset["schema"] == CLEAN_COLUMNS
        assert dataset["valid"] is True

    hourly = manifest["datasets"][0]
    assert hourly["start_timestamp"] == "2026-06-01T00:00:00Z"
    assert hourly["end_timestamp"] == "2026-06-01T02:00:00Z"

    quarter_hour = manifest["datasets"][1]
    assert quarter_hour["start_timestamp"] == "2026-06-01T00:00:00Z"
    assert quarter_hour["end_timestamp"] == "2026-06-01T00:30:00Z"


def test_refresh_does_not_write_manifest_when_a_timeframe_fails(tmp_path) -> None:
    manifest_dir = tmp_path / "manifests"

    with pytest.raises(CandleValidationError):
        refresh_candles(
            exchange_id="binance",
            symbol="BTC/USDT",
            timeframes=("1h", "15m"),
            limit=3,
            raw_dir=tmp_path / "raw",
            clean_dir=tmp_path / "clean",
            manifest_dir=manifest_dir,
            fetched_at=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
            now=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
            fetcher=failing_fetcher,
        )

    assert not list(manifest_dir.glob("*_manifest.json"))
