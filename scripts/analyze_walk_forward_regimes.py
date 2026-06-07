from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.evaluation import (  # noqa: E402
    DEFAULT_RETURN_REGIME_BUCKETS,
    EvaluationError,
    analyze_walk_forward_regimes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze walk-forward errors by absolute actual-return regime."
    )
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_RETURN_REGIME_BUCKETS)
    parser.add_argument("--output-name", default="walk_forward_regime_diagnostics.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = analyze_walk_forward_regimes(
            metrics=args.metrics,
            output_dir=args.output_dir,
            bucket_count=args.bucket_count,
            output_name=args.output_name,
        )
    except EvaluationError as exc:
        print(f"Walk-forward regime analysis failed: {exc}", file=sys.stderr)
        return 1

    print("metrics:")
    for path in run.metrics_paths:
        print(f"- {path}")
    print(f"regimes: {run.output_path}")
    print(f"regime rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
