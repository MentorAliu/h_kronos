from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hourly_prediction.config import (  # noqa: E402
    DEFAULT_EXCHANGE,
    DEFAULT_LIMIT,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAMES,
    SUPPORTED_TIMEFRAMES,
)
from hourly_prediction.refresh import refresh_candles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate the approved OHLCV timeframes in one run."
    )
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--timeframe",
        action="append",
        choices=sorted(SUPPORTED_TIMEFRAMES),
        help="Approved timeframe to refresh. Repeat for multiple. Defaults to 1h and 15m.",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--clean-dir", type=Path, default=ROOT / "data" / "clean")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeframes = tuple(args.timeframe or DEFAULT_TIMEFRAMES)
    results = refresh_candles(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframes=timeframes,
        limit=args.limit,
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
    )

    for result in results:
        print(f"{result.timeframe} raw: {result.raw_path}")
        print(f"{result.timeframe} clean: {result.clean_path}")
        print(f"{result.timeframe} clean rows: {result.clean_rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
