DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_LIMIT = 1000

SUPPORTED_TIMEFRAMES = {
    "15m": 15 * 60,
    "1h": 60 * 60,
}

CLEAN_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

RAW_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "exchange",
    "symbol",
    "timeframe",
    "fetched_at",
]
