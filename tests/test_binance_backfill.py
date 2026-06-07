from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.binance_backfill import (
    BinanceBackfillError,
    backfill_binance_klines,
    binance_monthly_klines_url,
    parse_binance_kline_zip,
)
from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.validation import CandleValidationError


def kline_zip(rows: list[list[object]], *, name: str = "fixture.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        text = "\n".join(",".join(str(value) for value in row) for row in rows) + "\n"
        archive.writestr(name, text)
    return buffer.getvalue()


def kline_row(open_time: int, *, close: float = 100.5) -> list[object]:
    return [
        open_time,
        100.0,
        101.0,
        99.0,
        close,
        10.0,
        open_time + 1,
        1000.0,
        1,
        5.0,
        500.0,
        0,
    ]


def test_binance_monthly_klines_url() -> None:
    assert binance_monthly_klines_url(
        symbol="BTCUSDT",
        timeframe="1h",
        month="2026-05",
    ) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-05.zip"
    )


def test_parse_binance_kline_zip_normalizes_millisecond_timestamps() -> None:
    raw = parse_binance_kline_zip(
        kline_zip(
            [
                kline_row(1_767_225_600_000, close=101.0),
                kline_row(1_767_229_200_000, close=102.0),
            ]
        ),
        exchange_id="binance_public",
        project_symbol="BTC/USDT",
        timeframe="1h",
        fetched_at=pd.Timestamp("2026-06-07T00:00:00Z"),
    )

    assert raw["timestamp"].tolist() == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
    ]
    assert raw["symbol"].tolist() == ["BTC/USDT", "BTC/USDT"]
    assert raw["timeframe"].tolist() == ["1h", "1h"]


def test_parse_binance_kline_zip_normalizes_microsecond_timestamps() -> None:
    raw = parse_binance_kline_zip(
        kline_zip([kline_row(1_767_225_600_000_000, close=101.0)]),
        exchange_id="binance_public",
        project_symbol="BTC/USDT",
        timeframe="1h",
        fetched_at=pd.Timestamp("2026-06-07T00:00:00Z"),
    )

    assert raw["timestamp"].tolist() == ["2026-01-01T00:00:00Z"]


def test_parse_binance_kline_zip_fails_on_mixed_timestamp_units() -> None:
    with pytest.raises(BinanceBackfillError, match="mixed timestamp units"):
        parse_binance_kline_zip(
            kline_zip(
                [
                    kline_row(1_767_225_600_000, close=101.0),
                    kline_row(1_767_229_200_000_000, close=102.0),
                ]
            ),
            exchange_id="binance_public",
            project_symbol="BTC/USDT",
            timeframe="1h",
            fetched_at=pd.Timestamp("2026-06-07T00:00:00Z"),
        )


def test_backfill_binance_klines_writes_multi_timeframe_manifest(tmp_path) -> None:
    fetched_at = pd.Timestamp("2026-06-07T00:00:00Z")

    def downloader(url: str) -> bytes:
        if "/1h/" in url:
            return kline_zip(
                [
                    kline_row(1_767_225_600_000, close=101.0),
                    kline_row(1_767_229_200_000, close=102.0),
                ]
            )
        return kline_zip(
            [
                kline_row(1_767_225_600_000, close=101.0),
                kline_row(1_767_226_500_000, close=102.0),
            ]
        )

    run = backfill_binance_klines(
        symbol="BTCUSDT",
        timeframes=("1h", "15m"),
        start_month="2026-01",
        end_month="2026-01",
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        manifest_dir=tmp_path / "manifests",
        fetched_at=fetched_at,
        now=pd.Timestamp("2026-06-07T00:00:00Z"),
        downloader=downloader,
    )

    assert run.manifest_path == (
        tmp_path / "manifests" / "binancepublic_BTCUSDT_20260607T000000Z_manifest.json"
    )
    assert [result.timeframe for result in run.results] == ["1h", "15m"]
    for result in run.results:
        assert result.raw_path.exists()
        assert result.clean_path.exists()
        clean = pd.read_csv(result.clean_path)
        assert list(clean.columns) == CLEAN_COLUMNS
        assert result.clean_rows == 2

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["exchange"] == "binance_public"
    assert manifest["symbol"] == "BTC/USDT"
    assert manifest["source_symbol"] == "BTCUSDT"
    assert manifest["start_month"] == "2026-01"
    assert manifest["end_month"] == "2026-01"
    assert manifest["timeframes"] == ["1h", "15m"]
    assert [dataset["timeframe"] for dataset in manifest["datasets"]] == ["1h", "15m"]


def test_backfill_binance_klines_fails_validation_without_manifest(tmp_path) -> None:
    def downloader(_: str) -> bytes:
        return kline_zip(
            [
                kline_row(1_767_225_600_000, close=101.0),
                kline_row(1_767_232_800_000, close=102.0),
            ]
        )

    with pytest.raises(CandleValidationError, match="gap or irregular"):
        backfill_binance_klines(
            symbol="BTCUSDT",
            timeframes=("1h",),
            start_month="2026-01",
            end_month="2026-01",
            raw_dir=tmp_path / "raw",
            clean_dir=tmp_path / "clean",
            manifest_dir=tmp_path / "manifests",
            fetched_at=pd.Timestamp("2026-06-07T00:00:00Z"),
            now=pd.Timestamp("2026-06-07T00:00:00Z"),
            downloader=downloader,
        )

    assert not list((tmp_path / "manifests").glob("*_manifest.json"))
