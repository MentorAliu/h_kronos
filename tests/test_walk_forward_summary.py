from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    WALK_FORWARD_SUMMARY_COLUMNS,
    EvaluationError,
    summarize_walk_forward_metrics,
)


def write_metrics(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timeframe": "1h",
                "kronos_absolute_error": 2.0,
                "kronos_squared_error": 4.0,
                "naive_absolute_error": 4.0,
                "naive_squared_error": 16.0,
                "kronos_direction_hit": True,
                "naive_direction_hit": False,
                "actual_return": 0.02,
                "forecasted_return": 0.01,
            },
            {
                "timeframe": "1h",
                "kronos_absolute_error": 4.0,
                "kronos_squared_error": 16.0,
                "naive_absolute_error": 2.0,
                "naive_squared_error": 4.0,
                "kronos_direction_hit": False,
                "naive_direction_hit": True,
                "actual_return": -0.01,
                "forecasted_return": 0.03,
            },
            {
                "timeframe": "15m",
                "kronos_absolute_error": 1.0,
                "kronos_squared_error": 1.0,
                "naive_absolute_error": 3.0,
                "naive_squared_error": 9.0,
                "kronos_direction_hit": True,
                "naive_direction_hit": False,
                "actual_return": 0.005,
                "forecasted_return": 0.006,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_summarize_walk_forward_metrics_writes_per_timeframe_summary(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = summarize_walk_forward_metrics(
        metrics=metrics_path,
        output_dir=tmp_path / "metrics",
    )

    assert result.rows == 2
    assert result.output_path == tmp_path / "metrics" / "run_walk_forward_summary.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == WALK_FORWARD_SUMMARY_COLUMNS
    one_hour = output.loc[output["timeframe"] == "1h"].iloc[0]
    assert one_hour["rows"] == 2
    assert one_hour["kronos_mae"] == pytest.approx(3.0)
    assert one_hour["kronos_rmse"] == pytest.approx(10.0**0.5)
    assert one_hour["naive_mae"] == pytest.approx(3.0)
    assert one_hour["naive_rmse"] == pytest.approx(10.0**0.5)
    assert one_hour["kronos_directional_accuracy"] == pytest.approx(0.5)
    assert one_hour["naive_directional_accuracy"] == pytest.approx(0.5)
    assert one_hour["average_actual_return"] == pytest.approx(0.005)
    assert one_hour["average_forecasted_return"] == pytest.approx(0.02)


def test_summarize_walk_forward_metrics_fails_on_missing_columns(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "bad_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timeframe": ["1h"]}).to_csv(metrics_path, index=False)

    with pytest.raises(EvaluationError, match="Walk-forward metrics missing required column"):
        summarize_walk_forward_metrics(
            metrics=metrics_path,
            output_dir=tmp_path / "metrics",
        )
