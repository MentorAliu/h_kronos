from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    WALK_FORWARD_EVALUATION_COLUMNS,
    EvaluationError,
    evaluate_kronos_walk_forward,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_kronos_walk_forward.py"


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
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
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


class FakePredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(
        self,
        *,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "df": df.copy(),
                "x_timestamp": x_timestamp.copy(),
                "y_timestamp": y_timestamp.copy(),
                "pred_len": pred_len,
                "top_p": kwargs["top_p"],
                "sample_count": kwargs["sample_count"],
            }
        )
        close = float(df["close"].iloc[-1]) + 1.0
        return pd.DataFrame(
            [
                {
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 10.0,
                    "amount": close * 10.0,
                }
            ]
        )


def test_walk_forward_uses_recent_windows_without_target_leakage(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 103.0, 102.0, 105.0, 104.0])
    write_manifest(manifest_path, clean_path=clean_path, run_id="wf_run")
    predictor = FakePredictor()

    result = evaluate_kronos_walk_forward(
        manifest=manifest_path,
        manifest_dir=tmp_path / "manifests",
        output_dir=tmp_path / "metrics",
        kronos_repo_path=tmp_path / "Kronos",
        device="cuda:0",
        lookback=2,
        max_windows=2,
        sma_window=2,
        top_p=0.7,
        sample_count=3,
        model_name="Fake/Kronos",
        tokenizer_name="Fake/Tokenizer",
        now=pd.Timestamp("2026-06-02T00:00:00Z"),
        predictor_loader=lambda **_: predictor,
    )

    assert result.rows == 2
    assert (
        result.output_path
        == tmp_path
        / "metrics"
        / "wf_run_kronos_top-p-0p7_sample-count-3_recent_walk_forward_metrics.csv"
    )
    assert len(predictor.calls) == 2
    for call in predictor.calls:
        target_timestamp = pd.Timestamp(call["y_timestamp"].iloc[0])
        assert call["pred_len"] == 1
        assert call["top_p"] == pytest.approx(0.7)
        assert call["sample_count"] == 3
        assert len(call["df"]) == 2
        assert pd.Timestamp(call["x_timestamp"].max()) < target_timestamp
        assert target_timestamp not in set(pd.to_datetime(call["x_timestamp"], utc=True))

    output = pd.read_csv(result.output_path)
    assert list(output.columns) == WALK_FORWARD_EVALUATION_COLUMNS
    assert output["forecast_timestamp"].tolist() == [
        "2026-06-01T04:00:00Z",
        "2026-06-01T05:00:00Z",
    ]
    assert output["model_name"].tolist() == ["Fake/Kronos", "Fake/Kronos"]
    assert output["top_p"].tolist() == [0.7, 0.7]
    assert output["sample_count"].tolist() == [3, 3]
    assert output["window_selection"].tolist() == ["recent", "recent"]
    assert output["current_close"].tolist() == [102.0, 105.0]
    assert output["target_close"].tolist() == [105.0, 104.0]
    assert output["kronos_close"].tolist() == [103.0, 106.0]
    assert output["kronos_close_error"].tolist() == [-2.0, 2.0]
    assert output["naive_close_error"].tolist() == [-3.0, 1.0]
    assert output["sma_close"].tolist() == [102.5, 103.5]
    assert output["sma_close_error"].tolist() == [-2.5, -0.5]
    assert output["sma_absolute_error"].tolist() == [2.5, 0.5]
    assert output["sma_squared_error"].tolist() == [6.25, 0.25]
    assert output["kronos_direction_hit"].tolist() == [True, False]
    assert output["naive_direction_hit"].tolist() == [False, False]
    assert output["sma_direction_hit"].tolist() == [True, True]


def test_walk_forward_requires_positive_max_windows(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 102.0])
    write_manifest(manifest_path, clean_path=clean_path)

    with pytest.raises(EvaluationError, match="max_windows must be positive"):
        evaluate_kronos_walk_forward(
            manifest=manifest_path,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "metrics",
            kronos_repo_path=tmp_path / "Kronos",
            lookback=2,
            max_windows=0,
            predictor_loader=lambda **_: FakePredictor(),
        )


def test_walk_forward_requires_valid_sampling_parameters(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 102.0])
    write_manifest(manifest_path, clean_path=clean_path)

    with pytest.raises(EvaluationError, match="top_p must be greater than 0 and at most 1"):
        evaluate_kronos_walk_forward(
            manifest=manifest_path,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "metrics",
            kronos_repo_path=tmp_path / "Kronos",
            lookback=2,
            top_p=0.0,
            predictor_loader=lambda **_: FakePredictor(),
        )

    with pytest.raises(EvaluationError, match="sample_count must be positive"):
        evaluate_kronos_walk_forward(
            manifest=manifest_path,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "metrics",
            kronos_repo_path=tmp_path / "Kronos",
            lookback=2,
            sample_count=0,
            predictor_loader=lambda **_: FakePredictor(),
        )


def test_walk_forward_even_selection_spans_valid_windows_deterministically(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(
        clean_path,
        closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
    )
    write_manifest(manifest_path, clean_path=clean_path, run_id="wf_run")

    result = evaluate_kronos_walk_forward(
        manifest=manifest_path,
        manifest_dir=tmp_path / "manifests",
        output_dir=tmp_path / "metrics",
        kronos_repo_path=tmp_path / "Kronos",
        lookback=2,
        max_windows=3,
        sma_window=2,
        window_selection="even",
        predictor_loader=lambda **_: FakePredictor(),
    )

    output = pd.read_csv(result.output_path)
    assert result.rows == 3
    assert output["forecast_timestamp"].tolist() == [
        "2026-06-01T02:00:00Z",
        "2026-06-01T04:00:00Z",
        "2026-06-01T07:00:00Z",
    ]
    assert output["window_number"].tolist() == [1, 2, 3]
    assert output["window_selection"].tolist() == ["even", "even", "even"]


def test_walk_forward_rejects_unknown_window_selection(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 102.0])
    write_manifest(manifest_path, clean_path=clean_path)

    with pytest.raises(EvaluationError, match="window_selection must be one of"):
        evaluate_kronos_walk_forward(
            manifest=manifest_path,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "metrics",
            kronos_repo_path=tmp_path / "Kronos",
            lookback=2,
            window_selection="middle",
            predictor_loader=lambda **_: FakePredictor(),
        )


def test_walk_forward_requires_lookback_at_least_sma_window(tmp_path) -> None:
    clean_path = tmp_path / "clean" / "candles.csv"
    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    write_clean(clean_path, closes=[100.0, 101.0, 102.0, 103.0])
    write_manifest(manifest_path, clean_path=clean_path)

    with pytest.raises(EvaluationError, match="lookback must be at least sma_window"):
        evaluate_kronos_walk_forward(
            manifest=manifest_path,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "metrics",
            kronos_repo_path=tmp_path / "Kronos",
            lookback=2,
            sma_window=3,
            predictor_loader=lambda **_: FakePredictor(),
        )


def test_walk_forward_cli_parses_sampling_args(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("evaluate_kronos_walk_forward_cli", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_kronos_walk_forward.py",
            "--kronos-repo-path",
            "vendor/Kronos",
            "--top-p",
            "0.7",
            "--sample-count",
            "3",
            "--window-selection",
            "even",
        ],
    )

    args = module.parse_args()

    assert args.top_p == pytest.approx(0.7)
    assert args.sample_count == 3
    assert args.window_selection == "even"
