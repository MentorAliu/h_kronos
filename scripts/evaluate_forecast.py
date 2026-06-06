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
    evaluate_forecast_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Kronos forecast rows against realized clean candles."
    )
    parser.add_argument("--forecast", type=Path, required=True)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = evaluate_forecast_manifest(
            forecast=args.forecast,
            manifest=args.manifest,
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
        )
    except EvaluationError as exc:
        print(f"Forecast evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"forecast: {run.forecast_path}")
    print(f"manifest: {run.manifest_path}")
    print(f"metrics: {run.output_path}")
    print(f"metric rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
