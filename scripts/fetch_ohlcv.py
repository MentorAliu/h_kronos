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
    SUPPORTED_TIMEFRAMES,
)
from hourly_prediction.data import fetch_ohlcv, raw_output_path, write_raw_candles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public OHLCV candles via CCXT.")
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", required=True, choices=sorted(SUPPORTED_TIMEFRAMES))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = fetch_ohlcv(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    )
    output_path = raw_output_path(
        output_dir=args.output_dir,
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    write_raw_candles(raw, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
