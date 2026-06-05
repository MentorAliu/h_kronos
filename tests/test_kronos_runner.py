from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.kronos_runner import (
    FORECAST_COLUMNS,
    KronosForecastError,
    resolve_manifest_path,
    run_kronos_forecast,
)


class FakePredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        *,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float,
        top_p: float,
        sample_count: int,
        verbose: bool,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "df": df.copy(),
                "x_timestamp": x_timestamp.copy(),
                "y_timestamp": y_timestamp.copy(),
                "pred_len": pred_len,
                "T": T,
                "top_p": top_p,
                "sample_count": sample_count,
                "verbose": verbose,
            }
        )
        return pd.DataFrame(
            {
                "open": [101.0],
                "high": [102.0],
                "low": [100.0],
                "close": [101.5],
                "volume": [20.0],
                "amount": [2030.0],
            },
            index=y_timestamp,
        )


def write_clean_csv(path: Path, *, timeframe: str, periods: int = 6) -> None:
    freq = "1h" if timeframe == "1h" else "15min"
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=periods, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": [100.0 + index for index in range(periods)],
            "high": [101.0 + index for index in range(periods)],
            "low": [99.0 + index for index in range(periods)],
            "close": [100.5 + index for index in range(periods)],
            "volume": [10.0 + index for index in range(periods)],
            "amount": [(100.5 + index) * (10.0 + index) for index in range(periods)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_manifest(path: Path, *, clean_dir: Path, run_id: str, periods: int = 6) -> Path:
    hourly_path = clean_dir / f"{run_id}_1h_clean.csv"
    quarter_hour_path = clean_dir / f"{run_id}_15m_clean.csv"
    write_clean_csv(hourly_path, timeframe="1h", periods=periods)
    write_clean_csv(quarter_hour_path, timeframe="15m", periods=periods)
    payload = {
        "run_id": run_id,
        "created_at": "2026-06-01T06:00:00Z",
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "limit": periods,
        "timeframes": ["1h", "15m"],
        "datasets": [
            {
                "timeframe": "1h",
                "clean_path": str(hourly_path),
                "clean_rows": periods,
                "schema": CLEAN_COLUMNS,
                "valid": True,
            },
            {
                "timeframe": "15m",
                "clean_path": str(quarter_hour_path),
                "clean_rows": periods,
                "schema": CLEAN_COLUMNS,
                "valid": True,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_resolve_manifest_path_accepts_explicit_path(tmp_path) -> None:
    manifest = write_manifest(
        tmp_path / "manifests" / "explicit_manifest.json",
        clean_dir=tmp_path / "clean",
        run_id="explicit",
    )

    assert resolve_manifest_path(manifest, manifest_dir=tmp_path / "manifests") == manifest


def test_resolve_manifest_path_latest_picks_newest(tmp_path) -> None:
    old_manifest = write_manifest(
        tmp_path / "manifests" / "old_manifest.json",
        clean_dir=tmp_path / "clean-old",
        run_id="old",
    )
    newest_manifest = write_manifest(
        tmp_path / "manifests" / "new_manifest.json",
        clean_dir=tmp_path / "clean-new",
        run_id="new",
    )
    old_time = pd.Timestamp("2026-06-01 00:00:00").timestamp()
    new_time = pd.Timestamp("2026-06-01 01:00:00").timestamp()
    old_manifest.touch()
    newest_manifest.touch()
    import os

    os.utime(old_manifest, (old_time, old_time))
    os.utime(newest_manifest, (new_time, new_time))

    assert resolve_manifest_path("latest", manifest_dir=tmp_path / "manifests") == newest_manifest


def test_run_kronos_forecast_uses_last_lookback_and_writes_contract(tmp_path) -> None:
    manifest = write_manifest(
        tmp_path / "manifests" / "run_manifest.json",
        clean_dir=tmp_path / "clean",
        run_id="binance_BTCUSDT_20260601T060000Z",
        periods=6,
    )
    predictor = FakePredictor()

    run = run_kronos_forecast(
        manifest=manifest,
        manifest_dir=tmp_path / "manifests",
        output_dir=tmp_path / "forecasts",
        kronos_repo_path=tmp_path / "Kronos",
        device="cuda:0",
        lookback=3,
        predictor_loader=lambda **_: predictor,
        now=pd.Timestamp("2026-06-01 07:00:00", tz="UTC"),
    )

    assert run.output_path == (
        tmp_path / "forecasts" / "binance_BTCUSDT_20260601T060000Z_kronos-small_forecast.csv"
    )
    output = pd.read_csv(run.output_path)
    assert list(output.columns) == FORECAST_COLUMNS
    assert output["timeframe"].tolist() == ["1h", "15m"]
    assert output["forecast_timestamp"].tolist() == [
        "2026-06-01T06:00:00Z",
        "2026-06-01T01:30:00Z",
    ]
    assert output["input_start_timestamp"].tolist() == [
        "2026-06-01T03:00:00Z",
        "2026-06-01T00:45:00Z",
    ]
    assert output["input_end_timestamp"].tolist() == [
        "2026-06-01T05:00:00Z",
        "2026-06-01T01:15:00Z",
    ]

    assert len(predictor.calls) == 2
    hourly_call = predictor.calls[0]
    assert hourly_call["df"]["close"].tolist() == [103.5, 104.5, 105.5]
    assert hourly_call["x_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-06-01T03:00:00Z",
        "2026-06-01T04:00:00Z",
        "2026-06-01T05:00:00Z",
    ]
    assert hourly_call["y_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-06-01T06:00:00Z"
    ]
    assert hourly_call["pred_len"] == 1
    assert hourly_call["verbose"] is False


def test_run_kronos_forecast_fails_when_clean_rows_are_short(tmp_path) -> None:
    manifest = write_manifest(
        tmp_path / "manifests" / "short_manifest.json",
        clean_dir=tmp_path / "clean",
        run_id="short",
        periods=2,
    )

    with pytest.raises(KronosForecastError, match="requires at least 3 clean rows"):
        run_kronos_forecast(
            manifest=manifest,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "forecasts",
            kronos_repo_path=tmp_path / "Kronos",
            device="cuda:0",
            lookback=3,
            predictor_loader=lambda **_: FakePredictor(),
        )
