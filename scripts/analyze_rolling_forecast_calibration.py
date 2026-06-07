from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.evaluation import (  # noqa: E402
    DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS,
    EvaluationError,
    analyze_rolling_forecast_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rolling leakage-safe return calibration for walk-forward forecasts."
    )
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold-bps", type=int, action="append", default=None)
    parser.add_argument(
        "--output-name",
        default="walk_forward_rolling_forecast_calibration.csv",
    )
    parser.add_argument(
        "--aggregate-output-name",
        default="walk_forward_rolling_forecast_calibration_aggregate.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = analyze_rolling_forecast_calibration(
            metrics=args.metrics,
            output_dir=args.output_dir,
            folds=args.folds,
            thresholds_bps=args.threshold_bps or DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS,
            output_name=args.output_name,
            aggregate_output_name=args.aggregate_output_name,
        )
    except EvaluationError as exc:
        print(f"Rolling forecast calibration analysis failed: {exc}", file=sys.stderr)
        return 1

    print("metrics:")
    for path in run.metrics_paths:
        print(f"- {path}")
    print(f"rolling forecast calibration: {run.output_path}")
    print(f"rolling forecast calibration rows: {run.rows}")
    print(f"rolling forecast calibration aggregate: {run.aggregate_output_path}")
    print(f"rolling forecast calibration aggregate rows: {run.aggregate_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
