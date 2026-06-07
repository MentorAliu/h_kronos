from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    FORECAST_CALIBRATION_COLUMNS,
    EvaluationError,
    analyze_forecast_calibration,
)


def write_metrics(path: Path, *, constant_forecast: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    actual_returns = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
    forecasted_returns = [0.01] * 6 if constant_forecast else [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for index, (actual_return, forecasted_return) in enumerate(
        zip(actual_returns, forecasted_returns, strict=True)
    ):
        current_close = 100.0 + index
        target_close = current_close * (1.0 + actual_return)
        kronos_close = current_close * (1.0 + forecasted_return)
        naive_close = current_close
        sma_return = 0.02
        sma_close = current_close * (1.0 + sma_return)
        rows.append(
            {
                "model_name": "Fake/Kronos",
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": "even",
                "timeframe": "1h",
                "forecast_timestamp": f"2026-06-01T0{index}:00:00Z",
                "current_close": current_close,
                "target_close": target_close,
                "kronos_close_error": kronos_close - target_close,
                "kronos_absolute_error": abs(kronos_close - target_close),
                "kronos_squared_error": (kronos_close - target_close) ** 2,
                "naive_close_error": naive_close - target_close,
                "naive_absolute_error": abs(naive_close - target_close),
                "naive_squared_error": (naive_close - target_close) ** 2,
                "sma_close_error": sma_close - target_close,
                "sma_absolute_error": abs(sma_close - target_close),
                "sma_squared_error": (sma_close - target_close) ** 2,
                "actual_return": actual_return,
                "forecasted_return": forecasted_return,
                "naive_return": 0.0,
                "sma_return": sma_return,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_analyze_forecast_calibration_uses_chronological_holdout(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        train_fraction=0.5,
    )

    assert result.rows == 1
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == FORECAST_CALIBRATION_COLUMNS
    row = output.iloc[0]
    assert row["train_rows"] == 3
    assert row["test_rows"] == 3
    assert row["input_transform"] == "raw"
    assert row["train_end_timestamp"] == "2026-06-01T02:00:00Z"
    assert row["test_start_timestamp"] == "2026-06-01T03:00:00Z"


def test_analyze_forecast_calibration_fits_train_only_and_scores_test_only(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        train_fraction=0.5,
    )

    row = pd.read_csv(result.output_path).iloc[0]
    assert row["bias_correction"] == pytest.approx(0.03)
    assert row["linear_alpha"] == pytest.approx(0.01)
    assert row["linear_beta"] == pytest.approx(2.0)
    assert not bool(row["linear_degenerate"])
    assert row["uncalibrated_return_mae"] == pytest.approx(
        (abs(0.04 - 0.09) + abs(0.05 - 0.11) + abs(0.06 - 0.13)) / 3
    )
    assert row["bias_return_mae"] == pytest.approx(
        (abs(0.07 - 0.09) + abs(0.08 - 0.11) + abs(0.09 - 0.13)) / 3
    )
    assert row["linear_return_mae"] == pytest.approx(0.0)
    assert row["uncalibrated_close_mae"] == pytest.approx((103 * 0.05 + 104 * 0.06 + 105 * 0.07) / 3)
    assert row["linear_close_mae"] == pytest.approx(0.0)
    assert row["uncalibrated_beats_naive_return_mae"]
    assert row["linear_beats_naive_return_mae"]


def test_analyze_forecast_calibration_handles_degenerate_linear_fit(tmp_path) -> None:
    metrics_path = write_metrics(
        tmp_path / "metrics" / "constant_walk_forward_metrics.csv",
        constant_forecast=True,
    )

    result = analyze_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        train_fraction=0.5,
    )

    row = pd.read_csv(result.output_path).iloc[0]
    assert bool(row["linear_degenerate"])
    assert row["linear_beta"] == 0.0
    assert row["linear_alpha"] == pytest.approx((0.03 + 0.05 + 0.07) / 3)


def test_analyze_forecast_calibration_reports_threshold_direction(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        train_fraction=0.5,
    )

    row = pd.read_csv(result.output_path).iloc[0]
    assert row["threshold_25_bps_rows"] == 3
    assert row["threshold_25_bps_uncalibrated_directional_accuracy"] == pytest.approx(1.0)
    assert row["threshold_25_bps_naive_directional_accuracy"] == pytest.approx(0.0)
    assert row["uncalibrated_beats_naive_any_threshold_direction"]


def test_analyze_forecast_calibration_fails_on_missing_columns(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "bad_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timeframe": ["1h"]}).to_csv(metrics_path, index=False)

    with pytest.raises(EvaluationError, match="Forecast calibration metrics missing required"):
        analyze_forecast_calibration(
            metrics=[metrics_path],
            output_dir=tmp_path / "metrics",
        )
