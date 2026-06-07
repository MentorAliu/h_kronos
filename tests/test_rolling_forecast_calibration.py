from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    EvaluationError,
    ROLLING_FORECAST_CALIBRATION_AGGREGATE_COLUMNS,
    ROLLING_FORECAST_CALIBRATION_COLUMNS,
    analyze_rolling_forecast_calibration,
)


def write_metrics(
    path: Path,
    *,
    model_name: str = "Fake/Kronos",
    timeframe: str = "1h",
    input_transform: str = "raw",
    constant_forecast: bool = False,
    rows: int = 8,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_returns = [0.02 + (0.01 * index) for index in range(rows)]
    forecasted_returns = [0.01] * rows if constant_forecast else [0.01 + (0.005 * index) for index in range(rows)]
    output_rows = []
    for index, (actual_return, forecasted_return) in enumerate(
        zip(actual_returns, forecasted_returns, strict=True)
    ):
        current_close = 100.0 + index
        target_close = current_close * (1.0 + actual_return)
        kronos_close = current_close * (1.0 + forecasted_return)
        naive_close = current_close
        sma_return = 0.015
        sma_close = current_close * (1.0 + sma_return)
        output_rows.append(
            {
                "model_name": model_name,
                "top_p": 0.9,
                "sample_count": 3 if model_name.endswith("small") else 1,
                "window_selection": "even",
                "input_transform": input_transform,
                "timeframe": timeframe,
                "forecast_timestamp": f"2026-06-01T{index:02d}:00:00Z",
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
    pd.DataFrame(output_rows).to_csv(path, index=False)
    return path


def test_analyze_rolling_forecast_calibration_reports_chronological_folds(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_rolling_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        folds=3,
    )

    assert result.rows == 3
    assert result.aggregate_rows == 1
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == ROLLING_FORECAST_CALIBRATION_COLUMNS
    assert output["fold"].tolist() == [1, 2, 3]
    assert output["train_rows"].tolist() == [2, 4, 6]
    assert output["test_rows"].tolist() == [2, 2, 2]
    assert output["train_end_timestamp"].tolist() == [
        "2026-06-01T01:00:00Z",
        "2026-06-01T03:00:00Z",
        "2026-06-01T05:00:00Z",
    ]
    assert output["test_start_timestamp"].tolist() == [
        "2026-06-01T02:00:00Z",
        "2026-06-01T04:00:00Z",
        "2026-06-01T06:00:00Z",
    ]


def test_analyze_rolling_forecast_calibration_fits_train_only(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_rolling_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        folds=3,
    )

    first = pd.read_csv(result.output_path).iloc[0]
    assert first["bias_correction"] == pytest.approx((0.01 + 0.015) / 2)
    assert first["linear_alpha"] == pytest.approx(0.0)
    assert first["linear_beta"] == pytest.approx(2.0)
    assert not bool(first["linear_degenerate"])
    assert first["uncalibrated_return_mae"] == pytest.approx((0.02 + 0.025) / 2)
    assert first["bias_return_mae"] == pytest.approx((0.0075 + 0.0125) / 2)
    assert first["linear_return_mae"] == pytest.approx(0.0)
    assert first["linear_close_mae"] == pytest.approx(0.0)


def test_analyze_rolling_forecast_calibration_handles_degenerate_linear_fit(tmp_path) -> None:
    metrics_path = write_metrics(
        tmp_path / "metrics" / "constant_walk_forward_metrics.csv",
        constant_forecast=True,
    )

    result = analyze_rolling_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        folds=3,
    )

    first = pd.read_csv(result.output_path).iloc[0]
    assert bool(first["linear_degenerate"])
    assert first["linear_beta"] == 0.0
    assert first["linear_alpha"] == pytest.approx((0.02 + 0.03) / 2)


def test_analyze_rolling_forecast_calibration_keeps_groups_separate(tmp_path) -> None:
    first_path = write_metrics(
        tmp_path / "metrics" / "small_raw_walk_forward_metrics.csv",
        model_name="Fake/Kronos-small",
        input_transform="raw",
    )
    second_path = write_metrics(
        tmp_path / "metrics" / "base_log_return_walk_forward_metrics.csv",
        model_name="Fake/Kronos-base",
        timeframe="15m",
        input_transform="log-return",
    )

    result = analyze_rolling_forecast_calibration(
        metrics=[first_path, second_path],
        output_dir=tmp_path / "metrics",
        folds=3,
    )

    output = pd.read_csv(result.output_path)
    aggregate = pd.read_csv(result.aggregate_output_path)
    assert result.rows == 6
    assert result.aggregate_rows == 2
    assert list(aggregate.columns) == ROLLING_FORECAST_CALIBRATION_AGGREGATE_COLUMNS
    assert output.groupby(
        ["model_name", "sample_count", "window_selection", "input_transform", "timeframe"]
    ).ngroups == 2
    assert set(aggregate["input_transform"]) == {"raw", "log-return"}


def test_analyze_rolling_forecast_calibration_aggregate_reports_fold_wins(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_rolling_forecast_calibration(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        folds=3,
    )

    aggregate = pd.read_csv(result.aggregate_output_path).iloc[0]
    assert aggregate["fold_count"] == 3
    assert aggregate["linear_beats_naive_return_mae_folds"] == 3
    assert aggregate["linear_beats_naive_close_mae_folds"] == 3
    assert bool(aggregate["linear_beats_naive_return_mae_all_folds"])
    assert bool(aggregate["linear_beats_naive_close_mae_all_folds"])
    assert aggregate["linear_vs_naive_return_mae_worst_delta"] <= 0


def test_analyze_rolling_forecast_calibration_fails_on_missing_columns(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "bad_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timeframe": ["1h"]}).to_csv(metrics_path, index=False)

    with pytest.raises(EvaluationError, match="Rolling forecast calibration metrics missing required"):
        analyze_rolling_forecast_calibration(
            metrics=[metrics_path],
            output_dir=tmp_path / "metrics",
        )


def test_analyze_rolling_forecast_calibration_fails_on_insufficient_rows(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "short_walk_forward_metrics.csv", rows=5)

    with pytest.raises(EvaluationError, match="requires at least 8 rows"):
        analyze_rolling_forecast_calibration(
            metrics=[metrics_path],
            output_dir=tmp_path / "metrics",
            folds=3,
        )
