from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.binance_backfill import (  # noqa: E402
    BinanceBackfillError,
    backfill_binance_klines,
)
from hourly_prediction.config import DEFAULT_TIMEFRAMES, SUPPORTED_TIMEFRAMES  # noqa: E402
from hourly_prediction.validation import CandleValidationError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Binance public monthly spot klines and validate clean candles."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--timeframe",
        action="append",
        choices=sorted(SUPPORTED_TIMEFRAMES),
        help="Approved timeframe to backfill. Repeat for multiple. Defaults to 1h and 15m.",
    )
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--clean-dir", type=Path, default=ROOT / "data" / "clean")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "outputs" / "manifests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeframes = tuple(args.timeframe or DEFAULT_TIMEFRAMES)
    try:
        run = backfill_binance_klines(
            symbol=args.symbol,
            timeframes=timeframes,
            start_month=args.start_month,
            end_month=args.end_month,
            raw_dir=args.output_dir,
            clean_dir=args.clean_dir,
            manifest_dir=args.manifest_dir,
        )
    except (BinanceBackfillError, CandleValidationError) as exc:
        print(f"Binance public backfill failed: {exc}", file=sys.stderr)
        return 1

    for result in run.results:
        print(f"{result.timeframe} raw: {result.raw_path}")
        print(f"{result.timeframe} clean: {result.clean_path}")
        print(f"{result.timeframe} clean rows: {result.clean_rows}")
        print(f"{result.timeframe} months: {result.months[0]}..{result.months[-1]}")
    print(f"manifest: {run.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
