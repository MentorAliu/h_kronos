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
    evaluate_baseline_manifest,
)
from hourly_prediction.kronos_runner import DEFAULT_LOOKBACK  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate naive close-persistence baseline metrics from a refresh manifest."
    )
    parser.add_argument("--manifest", default="latest")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "outputs" / "manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = evaluate_baseline_manifest(
            manifest=args.manifest,
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            lookback=args.lookback,
        )
    except EvaluationError as exc:
        print(f"Baseline evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"manifest: {run.manifest_path}")
    print(f"metrics: {run.output_path}")
    print(f"metric rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
