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
    analyze_forecast_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze leakage-safe return calibration for walk-forward forecasts."
    )
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--threshold-bps", type=int, action="append", default=None)
    parser.add_argument("--output-name", default="walk_forward_forecast_calibration.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = analyze_forecast_calibration(
            metrics=args.metrics,
            output_dir=args.output_dir,
            train_fraction=args.train_fraction,
            thresholds_bps=args.threshold_bps or DEFAULT_TARGET_FORMULATION_THRESHOLDS_BPS,
            output_name=args.output_name,
        )
    except EvaluationError as exc:
        print(f"Forecast calibration analysis failed: {exc}", file=sys.stderr)
        return 1

    print("metrics:")
    for path in run.metrics_paths:
        print(f"- {path}")
    print(f"forecast calibration: {run.output_path}")
    print(f"forecast calibration rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
