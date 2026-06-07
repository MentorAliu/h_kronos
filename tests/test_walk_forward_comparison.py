from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    WALK_FORWARD_COMPARISON_COLUMNS,
    EvaluationError,
    compare_walk_forward_reports,
)


def write_summary(
    path: Path,
    *,
    model_name: str,
    kronos_mae: float,
    naive_mae: float,
    window_selection: str = "even",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": window_selection,
                "timeframe": "1h",
                "rows": 100,
                "kronos_mae": kronos_mae,
                "kronos_rmse": kronos_mae + 1.0,
                "naive_mae": naive_mae,
                "naive_rmse": naive_mae + 1.0,
                "sma_mae": 8.0,
                "sma_rmse": 9.0,
                "kronos_directional_accuracy": 0.55,
                "naive_directional_accuracy": 0.0,
                "sma_directional_accuracy": 0.49,
                "average_actual_return": -0.001,
                "average_forecasted_return": 0.002,
            }
        ]
    ).to_csv(path, index=False)
    return path


def write_diagnostics(
    path: Path,
    *,
    model_name: str,
    kronos_vs_naive_mae_ratio: float,
    window_selection: str = "even",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": window_selection,
                "timeframe": "1h",
                "rows": 100,
                "random_seed": 42,
                "kronos_mean_signed_error": 1.5,
                "naive_mean_signed_error": 0.5,
                "sma_mean_signed_error": 3.0,
                "kronos_median_absolute_error": 4.0,
                "naive_median_absolute_error": 2.0,
                "sma_median_absolute_error": 7.0,
                "kronos_error_std": 6.0,
                "naive_error_std": 3.0,
                "sma_error_std": 10.0,
                "kronos_vs_naive_mae_delta": 2.0,
                "kronos_vs_naive_mae_ratio": kronos_vs_naive_mae_ratio,
                "kronos_vs_sma_mae_delta": -2.0,
                "kronos_vs_sma_mae_ratio": 0.75,
                "average_actual_return": -0.001,
                "average_forecasted_return": 0.002,
                "random_directional_accuracy": 0.28,
                "kronos_actual_up_pred_up": 1,
                "kronos_actual_up_pred_down": 2,
                "kronos_actual_up_pred_flat": 0,
                "kronos_actual_down_pred_up": 3,
                "kronos_actual_down_pred_down": 4,
                "kronos_actual_down_pred_flat": 0,
                "kronos_actual_flat_pred_up": 0,
                "kronos_actual_flat_pred_down": 0,
                "kronos_actual_flat_pred_flat": 0,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_compare_walk_forward_reports_joins_summary_and_diagnostics(tmp_path) -> None:
    small_summary = write_summary(
        tmp_path / "reports" / "small_summary.csv",
        model_name="Fake/Kronos-small",
        kronos_mae=6.0,
        naive_mae=4.0,
    )
    small_diag = write_diagnostics(
        tmp_path / "reports" / "small_diagnostics.csv",
        model_name="Fake/Kronos-small",
        kronos_vs_naive_mae_ratio=1.5,
    )
    base_summary = write_summary(
        tmp_path / "reports" / "base_summary.csv",
        model_name="Fake/Kronos-base",
        kronos_mae=3.0,
        naive_mae=4.0,
    )
    base_diag = write_diagnostics(
        tmp_path / "reports" / "base_diagnostics.csv",
        model_name="Fake/Kronos-base",
        kronos_vs_naive_mae_ratio=0.75,
    )

    result = compare_walk_forward_reports(
        summaries=[small_summary, base_summary],
        diagnostics=[small_diag, base_diag],
        output_dir=tmp_path / "metrics",
    )

    assert result.rows == 2
    assert result.output_path == tmp_path / "metrics" / "walk_forward_model_comparison.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == WALK_FORWARD_COMPARISON_COLUMNS
    base = output.loc[output["model_name"] == "Fake/Kronos-base"].iloc[0]
    assert base["timeframe"] == "1h"
    assert base["window_selection"] == "even"
    assert base["kronos_mae"] == pytest.approx(3.0)
    assert base["kronos_vs_naive_mae_ratio"] == pytest.approx(0.75)
    assert bool(base["beats_naive_mae"]) is True
    assert bool(base["beats_sma_mae"]) is True

    small = output.loc[output["model_name"] == "Fake/Kronos-small"].iloc[0]
    assert bool(small["beats_naive_mae"]) is False


def test_compare_walk_forward_reports_fails_on_mismatched_inputs(tmp_path) -> None:
    summary = write_summary(
        tmp_path / "reports" / "small_summary.csv",
        model_name="Fake/Kronos-small",
        kronos_mae=6.0,
        naive_mae=4.0,
    )
    diagnostic = write_diagnostics(
        tmp_path / "reports" / "base_diagnostics.csv",
        model_name="Fake/Kronos-small",
        kronos_vs_naive_mae_ratio=0.75,
        window_selection="recent",
    )

    with pytest.raises(EvaluationError, match="missing matching diagnostics"):
        compare_walk_forward_reports(
            summaries=[summary],
            diagnostics=[diagnostic],
            output_dir=tmp_path / "metrics",
        )
