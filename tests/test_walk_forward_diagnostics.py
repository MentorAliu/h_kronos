from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    WALK_FORWARD_DIAGNOSTIC_COLUMNS,
    EvaluationError,
    diagnose_walk_forward_metrics,
)


def write_metrics(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "Fake/Kronos-small",
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": "even",
                "timeframe": "1h",
                "kronos_close_error": -2.0,
                "kronos_absolute_error": 2.0,
                "naive_close_error": -4.0,
                "naive_absolute_error": 4.0,
                "sma_close_error": -3.0,
                "sma_absolute_error": 3.0,
                "actual_return": 0.02,
                "forecasted_return": 0.01,
            },
            {
                "model_name": "Fake/Kronos-small",
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": "even",
                "timeframe": "1h",
                "kronos_close_error": 4.0,
                "kronos_absolute_error": 4.0,
                "naive_close_error": 2.0,
                "naive_absolute_error": 2.0,
                "sma_close_error": 5.0,
                "sma_absolute_error": 5.0,
                "actual_return": -0.01,
                "forecasted_return": 0.03,
            },
            {
                "model_name": "Fake/Kronos-small",
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": "even",
                "timeframe": "1h",
                "kronos_close_error": 0.0,
                "kronos_absolute_error": 0.0,
                "naive_close_error": 1.0,
                "naive_absolute_error": 1.0,
                "sma_close_error": -1.0,
                "sma_absolute_error": 1.0,
                "actual_return": 0.0,
                "forecasted_return": -0.01,
            },
            {
                "model_name": "Fake/Kronos-base",
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": "recent",
                "timeframe": "15m",
                "kronos_close_error": 1.0,
                "kronos_absolute_error": 1.0,
                "naive_close_error": 3.0,
                "naive_absolute_error": 3.0,
                "sma_close_error": 2.0,
                "sma_absolute_error": 2.0,
                "actual_return": 0.005,
                "forecasted_return": 0.006,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_diagnose_walk_forward_metrics_writes_bias_and_confusion(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = diagnose_walk_forward_metrics(
        metrics=metrics_path,
        output_dir=tmp_path / "metrics",
        random_seed=42,
    )

    assert result.rows == 2
    assert result.output_path == tmp_path / "metrics" / "run_walk_forward_diagnostics.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == WALK_FORWARD_DIAGNOSTIC_COLUMNS

    one_hour = output.loc[
        (output["timeframe"] == "1h") & (output["model_name"] == "Fake/Kronos-small")
    ].iloc[0]
    assert one_hour["rows"] == 3
    assert one_hour["top_p"] == pytest.approx(0.9)
    assert one_hour["sample_count"] == 1
    assert one_hour["window_selection"] == "even"
    assert one_hour["kronos_mean_signed_error"] == pytest.approx(2.0 / 3.0)
    assert one_hour["naive_mean_signed_error"] == pytest.approx(-1.0 / 3.0)
    assert one_hour["sma_mean_signed_error"] == pytest.approx(1.0 / 3.0)
    assert one_hour["kronos_median_absolute_error"] == pytest.approx(2.0)
    assert one_hour["naive_median_absolute_error"] == pytest.approx(2.0)
    assert one_hour["sma_median_absolute_error"] == pytest.approx(3.0)
    assert one_hour["kronos_error_std"] == pytest.approx(np.std([-2.0, 4.0, 0.0], ddof=0))
    assert one_hour["kronos_vs_naive_mae_delta"] == pytest.approx(2.0 - (7.0 / 3.0))
    assert one_hour["kronos_vs_naive_mae_ratio"] == pytest.approx(2.0 / (7.0 / 3.0))
    assert one_hour["kronos_vs_sma_mae_delta"] == pytest.approx(2.0 - 3.0)
    assert one_hour["kronos_vs_sma_mae_ratio"] == pytest.approx(2.0 / 3.0)
    assert one_hour["kronos_actual_up_pred_up"] == 1
    assert one_hour["kronos_actual_down_pred_up"] == 1
    assert one_hour["kronos_actual_flat_pred_down"] == 1
    assert one_hour["average_actual_return"] == pytest.approx(0.01 / 3.0)
    assert one_hour["average_forecasted_return"] == pytest.approx(0.03 / 3.0)
    assert one_hour["random_directional_accuracy"] == pytest.approx(1 / 3)


def test_diagnose_walk_forward_metrics_random_baseline_is_deterministic(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    first = diagnose_walk_forward_metrics(
        metrics=metrics_path,
        output_dir=tmp_path / "first",
        random_seed=7,
    )
    second = diagnose_walk_forward_metrics(
        metrics=metrics_path,
        output_dir=tmp_path / "second",
        random_seed=7,
    )

    assert pd.read_csv(first.output_path)["random_directional_accuracy"].tolist() == pd.read_csv(
        second.output_path
    )["random_directional_accuracy"].tolist()


def test_diagnose_walk_forward_metrics_fails_on_missing_columns(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "bad_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timeframe": ["1h"]}).to_csv(metrics_path, index=False)

    with pytest.raises(EvaluationError, match="Walk-forward diagnostics missing required column"):
        diagnose_walk_forward_metrics(
            metrics=metrics_path,
            output_dir=tmp_path / "metrics",
        )
