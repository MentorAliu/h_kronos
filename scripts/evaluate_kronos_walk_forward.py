from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.evaluation import (  # noqa: E402
    DEFAULT_MAX_WALK_FORWARD_WINDOWS,
    DEFAULT_SMA_WINDOW,
    DEFAULT_WINDOW_SELECTION,
    EvaluationError,
    SUPPORTED_WINDOW_SELECTIONS,
    evaluate_kronos_walk_forward,
)
from hourly_prediction.kronos_runner import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_LOOKBACK,
    DEFAULT_PRED_LEN,
)
from hourly_prediction.kronos_runtime import (  # noqa: E402
    DEFAULT_KRONOS_MODEL,
    DEFAULT_KRONOS_TOKENIZER,
    KronosRuntimeError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded walk-forward Kronos evaluation against naive close persistence."
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
    parser.add_argument("--kronos-repo-path", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WALK_FORWARD_WINDOWS)
    parser.add_argument("--sma-window", type=int, default=DEFAULT_SMA_WINDOW)
    parser.add_argument("--pred-len", type=int, default=DEFAULT_PRED_LEN)
    parser.add_argument("--model-name", default=DEFAULT_KRONOS_MODEL)
    parser.add_argument("--tokenizer-name", default=DEFAULT_KRONOS_TOKENIZER)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument(
        "--window-selection",
        choices=SUPPORTED_WINDOW_SELECTIONS,
        default=DEFAULT_WINDOW_SELECTION,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = evaluate_kronos_walk_forward(
            manifest=args.manifest,
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            kronos_repo_path=args.kronos_repo_path,
            device=args.device,
            lookback=args.lookback,
            max_windows=args.max_windows,
            sma_window=args.sma_window,
            pred_len=args.pred_len,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            top_p=args.top_p,
            sample_count=args.sample_count,
            window_selection=args.window_selection,
        )
    except (EvaluationError, KronosRuntimeError) as exc:
        print(f"Walk-forward evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"manifest: {run.manifest_path}")
    print(f"metrics: {run.output_path}")
    print(f"metric rows: {run.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
