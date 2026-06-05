from __future__ import annotations

import pandas as pd

from hourly_prediction.config import CLEAN_COLUMNS
from hourly_prediction.refresh import refresh_candles


def stub_fetcher(
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    freq = "1h" if timeframe == "1h" else "15min"
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=limit, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": [100.0 + i for i in range(limit)],
            "high": [101.0 + i for i in range(limit)],
            "low": [99.0 + i for i in range(limit)],
            "close": [100.5 + i for i in range(limit)],
            "volume": [10.0 + i for i in range(limit)],
            "exchange": exchange_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "fetched_at": "2026-06-01T05:00:00Z",
        }
    )


def test_refresh_candles_fetches_validates_and_writes_both_timeframes(tmp_path) -> None:
    results = refresh_candles(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframes=("1h", "15m"),
        limit=3,
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        fetched_at=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
        now=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
        fetcher=stub_fetcher,
    )

    assert [result.timeframe for result in results] == ["1h", "15m"]

    for result in results:
        assert result.raw_path.exists()
        assert result.clean_path.exists()
        clean = pd.read_csv(result.clean_path)
        assert list(clean.columns) == CLEAN_COLUMNS
        assert result.clean_rows == 3
