# Kronos Data Ingestion Notes

## Goal

Use Kronos as a structured financial time-series model for OHLCV/K-line forecasting. Kronos should receive raw market candles, not chart screenshots.

## Recommended Pipeline

```text
exchange OHLCV data -> normalize to Kronos input shape -> Kronos forecast -> signal/backtest
```

Avoid screenshot/chart extraction unless raw data is unavailable.

## Best Free Data Path

Use **CCXT** as the ingestion adapter and **Binance klines / Binance public data** as the first data source when the pair is available.

Reasons:

- CCXT is free and open-source.
- CCXT normalizes exchange APIs behind `fetch_ohlcv`.
- Binance spot klines provide direct `open`, `high`, `low`, `close`, and `volume`.
- Binance public data / `data.binance.vision` is better for large historical backfills than paginating live REST API calls.

Fallbacks:

- Coinbase Exchange via CCXT for US-accessible pairs.
- Kraken via CCXT for recent OHLC data, though REST history is limited.

## Kronos Input Shape

Kronos expects a pandas DataFrame with at least:

```text
open, high, low, close
```

Recommended columns:

```text
open, high, low, close, volume, amount
```

Where:

```text
amount = close * volume
```

Keep timestamps as a separate pandas Series aligned to the candle rows.

## Timeframe Plan

Start with **1h** and **15m** next-candle prediction. Do not target a 24h forecast horizon in the initial build.

```text
Hourly model: 1h candles
lookback: 400-512 bars
pred_len: 1 bar
coverage: about 17-21 days
target: next 1h candle
```

```text
Intraday model: 15m candles
lookback: 400-512 bars
pred_len: 1 bar
coverage: about 4-5 days
target: next 15m candle
```

## Timeframe Tradeoffs

| Timeframe | 512-Bar Coverage | Best Use |
| --- | ---: | --- |
| 5m | ~42 hours | Short intraday moves, noisy |
| 15m | ~5 days | Intraday trend and timing |
| 1h | ~21 days | Hourly prediction |

## Initial Recommendation

Build the first baseline with:

```text
exchange: Binance through CCXT
symbol: BTC/USDT
timeframe: 1h
lookback: 512
pred_len: 1
```

Then build the matching 15m baseline:

```text
exchange: Binance through CCXT
symbol: BTC/USDT
timeframe: 15m
lookback: 512
pred_len: 1
```

Do not start with `1m`, `4h`, or 24h horizons. They add noise, storage, and modeling complexity before the core 1h/15m next-candle baselines are proven.
