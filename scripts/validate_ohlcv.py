from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.config import SUPPORTED_TIMEFRAMES  # noqa: E402
from hourly_prediction.validation import validate_candles, write_clean_candles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw OHLCV candles.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--timeframe", required=True, choices=sorted(SUPPORTED_TIMEFRAMES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "clean")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = pd.read_csv(args.input)
    clean = validate_candles(raw, timeframe=args.timeframe)
    output_path = args.output_dir / f"{args.input.stem.removesuffix('_raw')}_clean.csv"
    write_clean_candles(clean, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
