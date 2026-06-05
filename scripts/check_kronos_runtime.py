from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.kronos_runtime import (  # noqa: E402
    DEFAULT_KRONOS_MODEL,
    DEFAULT_KRONOS_TOKENIZER,
    DEFAULT_MAX_CONTEXT,
    KronosRuntimeError,
    check_kronos_runtime,
    print_runtime_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the local CUDA-first Kronos runtime before forecasting."
    )
    parser.add_argument("--kronos-repo-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("cuda", "model"), default="cuda")
    parser.add_argument("--model-name", default=DEFAULT_KRONOS_MODEL)
    parser.add_argument("--tokenizer-name", default=DEFAULT_KRONOS_TOKENIZER)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = check_kronos_runtime(
            kronos_repo_path=args.kronos_repo_path,
            device=args.device,
            mode=args.mode,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            max_context=args.max_context,
        )
    except KronosRuntimeError as exc:
        print(f"Kronos runtime check failed: {exc}", file=sys.stderr)
        return 1

    print_runtime_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
