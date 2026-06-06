from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    METRIC_COLUMNS,
    EvaluationError,
    FORECAST_EVALUATION_COLUMNS,
    build_evaluation_windows,
    evaluate_baseline_manifest,
    evaluate_forecast_manifest,
    naive_close_persistence_metrics,
)


def clean_candles(*, closes: list[float], timeframe: str = "1h") -> pd.DataFrame:
    freq = "1h" if timeframe == "1h" else "15min"
    timestamps = pd.date_range(
        "2026-06-01 00:00:00",
        periods=len(closes),
        freq=freq,
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": closes,
            "high": [close + 1 for close in closes],
            "low": [close - 1 for close in closes],
            "close": closes,
            "volume": [10.0 for _ in closes],
            "amount": [close * 10.0 for close in closes],
        }
    )


def write_clean(path: Path, *, closes: list[float], timeframe: str = "1h") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_candles(closes=closes, timeframe=timeframe).to_csv(path, index=False)


def write_manifest(path: Path, *, clean_path: Path, run_id: str = "run") -> Path:
    payload = {
        "run_id": run_id,
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "datasets": [
            {
                "timeframe": "1h",
                "clean_path": str(clean_path),
                "valid": True,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_forecast(path: Path, *, run_id: str = "forecast_run") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "forecast_created_at": "2026-06-01T02:05:00Z",
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "model_name": "NeoQuasar/Kronos-small",
                "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
                "device": "cuda:0",
                "lookback": 2,
                "pred_len": 1,
                "input_start_timestamp": "2026-06-01T00:00:00Z",
                "input_end_timestamp": "2026-06-01T01:00:00Z",
                "forecast_timestamp": "2026-06-01T02:00:00Z",
                "open": 102.0,
                "high": 104.0,
                "low": 101.0,
                "close": 103.0,
                "volume": 11.0,
                "amount": 1133.0,
            },
            {
                "run_id": run_id,
                "forecast_created_at": "2026-06-01T03:05:00Z",
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "model_name": "NeoQuasar/Kronos-small",
                "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
                "device": "cuda:0",
                "lookback": 2,
                "pred_len": 1,
                "input_start_timestamp": "2026-06-01T01:00:00Z",
                "input_end_timestamp": "2026-06-01T02:00:00Z",
                "forecast_timestamp": "2026-06-01T03:00:00Z",
                "open": 101.0,
                "high": 102.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 12.0,
                "amount": 1200.0,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_build_evaluation_windows_never_includes_target() -> None:
    clean = clean_candles(closes=[100.0, 101.0, 103.0, 102.0, 105.0])

    windows = build_evaluation_windows(clean, timeframe="1h", lookback=3)

    assert len(windows) == 2
    first = windows[0]
    assert first.input_window["close"].tolist() == [100.0, 101.0, 103.0]
    assert first.target_close == 102.0
    assert first.input_start_timestamp == "2026-06-01T00:00:00Z"
    assert first.input_end_timestamp == "2026-06-01T02:00:00Z"
    assert first.target_timestamp == "2026-06-01T03:00:00Z"

    for window in windows:
        assert window.input_window["timestamp"].max() < pd.Timestamp(window.target_timestamp)


def test_build_evaluation_windows_requires_lookback_plus_target() -> None:
    clean = clean_candles(closes=[100.0, 101.0, 102.0])

    with pytest.raises(EvaluationError, match="requires at least 4 clean rows"):
        build_evaluation_windows(clean, timeframe="1h", lookback=3)


def test_naive_close_persistence_metrics() -> None:
    clean = clean_candles(closes=[100.0, 102.0, 102.0, 101.0, 103.0])
    windows = build_evaluation_windows(clean, timeframe="1h", lookback=2)

    metrics = naive_close_persistence_metrics(
        windows,
        timeframe="1h",
        lookback=2,
    )

    assert metrics["timeframe"] == "1h"
    assert metrics["windows"] == 3
    assert metrics["lookback"] == 2
    assert metrics["mae"] == pytest.approx((0.0 + 1.0 + 2.0) / 3)
    assert metrics["rmse"] == pytest.approx(((0.0**2 + 1.0**2 + 2.0**2) / 3) ** 0.5)
    assert metrics["directional_accuracy"] == pytest.approx(1 / 3)


def test_evaluate_baseline_manifest_writes_metrics_csv(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 103.0, 102.0, 105.0])
    write_manifest(manifest_path, clean_path=clean_path, run_id="baseline_run")

    result = evaluate_baseline_manifest(
        manifest=manifest_path,
        manifest_dir=tmp_path / "manifests",
        output_dir=tmp_path / "metrics",
        lookback=3,
    )

    assert result.output_path == tmp_path / "metrics" / "baseline_run_naive_baseline_metrics.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == METRIC_COLUMNS
    assert output["timeframe"].tolist() == ["1h"]
    assert output["windows"].tolist() == [2]


def test_evaluate_forecast_manifest_matches_targets_and_writes_metrics(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    forecast_path = tmp_path / "forecasts" / "forecast.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 105.0, 99.0])
    write_manifest(manifest_path, clean_path=clean_path, run_id="realized_run")
    write_forecast(forecast_path)

    result = evaluate_forecast_manifest(
        forecast=forecast_path,
        manifest=manifest_path,
        manifest_dir=tmp_path / "manifests",
        output_dir=tmp_path / "metrics",
    )

    assert result.output_path == tmp_path / "metrics" / "forecast_run_forecast_metrics.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == FORECAST_EVALUATION_COLUMNS
    assert output["timeframe"].tolist() == ["1h", "1h"]
    assert output["target_close"].tolist() == [105.0, 99.0]
    assert output["kronos_close_error"].tolist() == [-2.0, 1.0]
    assert output["kronos_absolute_error"].tolist() == [2.0, 1.0]
    assert output["kronos_squared_error"].tolist() == [4.0, 1.0]
    assert output["naive_close"].tolist() == [101.0, 105.0]
    assert output["naive_close_error"].tolist() == [-4.0, 6.0]
    assert output["kronos_direction_hit"].tolist() == [True, True]
    assert output["naive_direction_hit"].tolist() == [False, False]
    assert output["actual_return"].tolist() == pytest.approx([4.0 / 101.0, -6.0 / 105.0])
    assert output["forecasted_return"].tolist() == pytest.approx([2.0 / 101.0, -5.0 / 105.0])
    assert output["naive_return"].tolist() == [0.0, 0.0]


def test_evaluate_forecast_manifest_fails_when_target_is_missing(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    forecast_path = tmp_path / "forecasts" / "forecast.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0])
    write_manifest(manifest_path, clean_path=clean_path)
    write_forecast(forecast_path)

    with pytest.raises(EvaluationError, match="No realized 1h candle found for forecast timestamp"):
        evaluate_forecast_manifest(
            forecast=forecast_path,
            manifest=manifest_path,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "metrics",
        )
