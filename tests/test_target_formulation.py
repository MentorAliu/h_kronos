from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    TARGET_FORMULATION_COLUMNS,
    EvaluationError,
    analyze_target_formulation,
)


def write_metrics(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [
        ("2026-06-01T00:00:00Z", 0.020, 0.010, 0.000, 0.015),
        ("2026-06-01T01:00:00Z", -0.010, -0.020, 0.000, -0.005),
        ("2026-06-01T02:00:00Z", 0.0003, -0.002, 0.000, 0.001),
    ]
    for model_name in ("Fake/Kronos-small", "Fake/Kronos-base"):
        for index, (timestamp, actual_return, forecasted_return, naive_return, sma_return) in enumerate(
            cases
        ):
            current_close = 100.0
            target_close = current_close * (1.0 + actual_return)
            kronos_close = current_close * (1.0 + forecasted_return)
            naive_close = current_close * (1.0 + naive_return)
            sma_close = current_close * (1.0 + sma_return)
            kronos_error = kronos_close - target_close
            naive_error = naive_close - target_close
            sma_error = sma_close - target_close
            rows.append(
                {
                    "model_name": model_name,
                    "top_p": 0.9,
                    "sample_count": 3 if model_name.endswith("small") else 1,
                    "window_selection": "even",
                    "timeframe": "1h",
                    "forecast_timestamp": timestamp,
                    "current_close": current_close,
                    "target_close": target_close,
                    "kronos_close_error": kronos_error,
                    "kronos_absolute_error": abs(kronos_error),
                    "kronos_squared_error": kronos_error**2,
                    "naive_close_error": naive_error,
                    "naive_absolute_error": abs(naive_error),
                    "naive_squared_error": naive_error**2,
                    "sma_close_error": sma_error,
                    "sma_absolute_error": abs(sma_error),
                    "sma_squared_error": sma_error**2,
                    "actual_return": actual_return,
                    "forecasted_return": forecasted_return,
                    "naive_return": naive_return,
                    "sma_return": sma_return,
                    "kronos_direction_hit": actual_return * forecasted_return > 0,
                    "naive_direction_hit": False,
                    "sma_direction_hit": actual_return * sma_return > 0,
                    "window_number": index + 1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_analyze_target_formulation_reports_return_error_and_bias(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_target_formulation(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
    )

    assert result.rows == 2
    assert result.output_path == tmp_path / "metrics" / "walk_forward_target_formulation.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == TARGET_FORMULATION_COLUMNS

    small = output.loc[output["model_name"] == "Fake/Kronos-small"].iloc[0]
    assert small["sample_count"] == 3
    assert small["window_selection"] == "even"
    assert small["input_transform"] == "raw"
    assert small["kronos_return_mae"] == pytest.approx(
        (abs(0.010 - 0.020) + abs(-0.020 - -0.010) + abs(-0.002 - 0.0003)) / 3
    )
    assert small["kronos_return_rmse"] == pytest.approx(
        (((0.010 - 0.020) ** 2 + (-0.020 - -0.010) ** 2 + (-0.002 - 0.0003) ** 2) / 3)
        ** 0.5
    )
    assert small["kronos_mean_signed_return_error"] == pytest.approx(
        ((0.010 - 0.020) + (-0.020 - -0.010) + (-0.002 - 0.0003)) / 3
    )
    assert small["kronos_median_signed_return_error"] == pytest.approx(-0.01)
    assert small["kronos_mean_signed_close_error"] == pytest.approx(
        ((101.0 - 102.0) + (98.0 - 99.0) + (99.8 - 100.03)) / 3
    )
    assert small["kronos_beats_naive_close_mae"]
    assert small["kronos_beats_naive_return_mae"]


def test_analyze_target_formulation_thresholds_exclude_low_move_rows(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_target_formulation(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
    )

    output = pd.read_csv(result.output_path)
    small = output.loc[output["model_name"] == "Fake/Kronos-small"].iloc[0]
    assert small["threshold_0_bps_rows"] == 3
    assert small["threshold_0_bps_kronos_directional_accuracy"] == pytest.approx(2 / 3)
    assert small["threshold_10_bps_rows"] == 2
    assert small["threshold_10_bps_kronos_directional_accuracy"] == pytest.approx(1.0)
    assert small["threshold_10_bps_naive_directional_accuracy"] == pytest.approx(0.0)
    assert small["threshold_10_bps_sma_directional_accuracy"] == pytest.approx(1.0)
    assert small["kronos_beats_naive_threshold_10_bps_direction"]


def test_analyze_target_formulation_keeps_configs_separate(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_target_formulation(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
    )

    output = pd.read_csv(result.output_path)
    assert sorted(output["model_name"].unique().tolist()) == [
        "Fake/Kronos-base",
        "Fake/Kronos-small",
    ]
    assert output.groupby(
        ["model_name", "sample_count", "window_selection", "input_transform", "timeframe"]
    ).ngroups == 2


def test_analyze_target_formulation_fails_on_missing_columns(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "bad_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timeframe": ["1h"]}).to_csv(metrics_path, index=False)

    with pytest.raises(EvaluationError, match="Target formulation metrics missing required"):
        analyze_target_formulation(
            metrics=[metrics_path],
            output_dir=tmp_path / "metrics",
        )
