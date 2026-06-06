from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.evaluation import (  # noqa: E402
    EvaluationError,
    summarize_walk_forward_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize row-level walk-forward Kronos metrics by timeframe."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = summarize_walk_forward_metrics(
            metrics=args.metrics,
            output_dir=args.output_dir,
        )
    except EvaluationError as exc:
        print(f"Walk-forward summary failed: {exc}", file=sys.stderr)
        return 1

    print(f"metrics: {run.metrics_path}")
    print(f"summary: {run.output_path}")
    print(f"summary rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
