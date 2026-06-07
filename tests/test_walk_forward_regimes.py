from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hourly_prediction.evaluation import (
    WALK_FORWARD_REGIME_COLUMNS,
    EvaluationError,
    analyze_walk_forward_regimes,
)


def write_metrics(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    returns = [-0.04, -0.02, 0.01, 0.03]
    forecasted = [-0.03, 0.01, 0.02, 0.04]
    for model_name in ("Fake/Kronos-small", "Fake/Kronos-base"):
        for index, (actual_return, forecasted_return) in enumerate(
            zip(returns, forecasted, strict=True)
        ):
            current_close = 100.0
            target_close = current_close * (1.0 + actual_return)
            kronos_close = current_close * (1.0 + forecasted_return)
            naive_close = current_close
            sma_close = current_close * 1.01
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
                    "forecast_timestamp": f"2026-06-01T0{index}:00:00Z",
                    "kronos_absolute_error": abs(kronos_error),
                    "kronos_squared_error": kronos_error**2,
                    "naive_absolute_error": abs(naive_error),
                    "naive_squared_error": naive_error**2,
                    "sma_absolute_error": abs(sma_error),
                    "sma_squared_error": sma_error**2,
                    "kronos_direction_hit": actual_return * forecasted_return > 0,
                    "naive_direction_hit": False,
                    "sma_direction_hit": actual_return > 0,
                    "actual_return": actual_return,
                    "forecasted_return": forecasted_return,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_analyze_walk_forward_regimes_buckets_by_target_return(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_walk_forward_regimes(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        bucket_count=2,
    )

    assert result.rows == 4
    assert result.output_path == tmp_path / "metrics" / "walk_forward_regime_diagnostics.csv"
    output = pd.read_csv(result.output_path)
    assert list(output.columns) == WALK_FORWARD_REGIME_COLUMNS
    assert set(output["return_regime"]) == {"q1_of_2_abs_return", "q2_of_2_abs_return"}

    small_low = output.loc[
        (output["model_name"] == "Fake/Kronos-small")
        & (output["return_regime"] == "q1_of_2_abs_return")
    ].iloc[0]
    assert small_low["rows"] == 2
    assert small_low["sample_count"] == 3
    assert small_low["window_selection"] == "even"
    assert small_low["input_transform"] == "raw"
    assert small_low["average_absolute_actual_return"] == pytest.approx(0.015)
    assert small_low["forecast_actual_return_correlation"] == pytest.approx(1.0)
    assert small_low["kronos_vs_naive_mae_ratio"] == pytest.approx(
        small_low["kronos_mae"] / small_low["naive_mae"]
    )


def test_analyze_walk_forward_regimes_keeps_configs_separate(tmp_path) -> None:
    metrics_path = write_metrics(tmp_path / "metrics" / "run_walk_forward_metrics.csv")

    result = analyze_walk_forward_regimes(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        bucket_count=2,
    )

    output = pd.read_csv(result.output_path)
    assert sorted(output["model_name"].unique().tolist()) == [
        "Fake/Kronos-base",
        "Fake/Kronos-small",
    ]
    assert output.groupby(["model_name", "sample_count", "window_selection", "input_transform"]).ngroups == 2


def test_analyze_walk_forward_regimes_marks_undefined_correlation(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "flat_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "Fake/Kronos",
                "top_p": 0.9,
                "sample_count": 1,
                "window_selection": "even",
                "timeframe": "15m",
                "forecast_timestamp": f"2026-06-01T00:0{index}:00Z",
                "kronos_absolute_error": 1.0,
                "kronos_squared_error": 1.0,
                "naive_absolute_error": 2.0,
                "naive_squared_error": 4.0,
                "sma_absolute_error": 3.0,
                "sma_squared_error": 9.0,
                "kronos_direction_hit": True,
                "naive_direction_hit": False,
                "sma_direction_hit": True,
                "actual_return": 0.01,
                "forecasted_return": 0.02 + index,
            }
            for index in range(2)
        ]
    ).to_csv(metrics_path, index=False)

    result = analyze_walk_forward_regimes(
        metrics=[metrics_path],
        output_dir=tmp_path / "metrics",
        bucket_count=1,
    )

    output = pd.read_csv(result.output_path)
    assert pd.isna(output["forecast_actual_return_correlation"].iloc[0])


def test_analyze_walk_forward_regimes_fails_on_missing_columns(tmp_path) -> None:
    metrics_path = tmp_path / "metrics" / "bad_walk_forward_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timeframe": ["1h"]}).to_csv(metrics_path, index=False)

    with pytest.raises(EvaluationError, match="Walk-forward regime metrics missing required"):
        analyze_walk_forward_regimes(
            metrics=[metrics_path],
            output_dir=tmp_path / "metrics",
        )
