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
    compare_walk_forward_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple walk-forward summary and diagnostics reports."
    )
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--diagnostics", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--output-name", default="walk_forward_model_comparison.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = compare_walk_forward_reports(
            summaries=args.summary,
            diagnostics=args.diagnostics,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
    except EvaluationError as exc:
        print(f"Walk-forward comparison failed: {exc}", file=sys.stderr)
        return 1

    print("summaries:")
    for path in run.summary_paths:
        print(f"- {path}")
    print("diagnostics:")
    for path in run.diagnostic_paths:
        print(f"- {path}")
    print(f"comparison: {run.output_path}")
    print(f"comparison rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
