# LLM Implementation Roadmap

## Purpose

This roadmap tells an LLM coding agent how to build the Kronos forecasting pipeline in small, verifiable steps.

The current initial target is:

```text
BTC/USDT raw OHLCV -> 1h and 15m clean datasets -> Kronos next-candle forecasts -> baseline evaluation
```

Do not implement 24h forecasts, `4h` regime filters, `1m` candles, or live trading until the initial `1h` and `15m` next-candle baselines are working and evaluated.

## Operating Principles

Build this as a reproducible research pipeline first.

- Keep each stage separate: fetch, validate, forecast, evaluate, backtest.
- Save artifacts to disk so results can be inspected later.
- Add tests before the logic becomes hard to reason about.
- Treat future leakage as the primary risk.
- Prefer simple, boring data contracts over clever abstractions.
- Do not let model output imply profitability without baseline and backtest evidence.

## Data Contract

Each clean candle dataset must have these columns:

```text
timestamp, open, high, low, close, volume, amount
```

Where:

```text
timestamp = candle open timestamp in UTC
amount = close * volume
```

Rules:

- Timestamps must be sorted ascending.
- Timestamps must be unique.
- Only closed candles are allowed.
- Candle intervals must be gap-checked.
- Numeric fields must be parseable as finite numbers.
- Clean datasets should be saved separately from raw downloads.

## Initial Directory Plan

```text
AGENTS.md
requirements.txt
docs/
  kronos-data-ingestion-notes.md
  llm-implementation-roadmap.md
scripts/
  fetch_ohlcv.py
  validate_ohlcv.py
  run_kronos_forecast.py
  evaluate_forecast.py
src/
  hourly_prediction/
    __init__.py
    config.py
    data.py
    validation.py
    kronos_runner.py
    evaluation.py
tests/
  test_validation.py
  test_evaluation_windows.py
data/
  raw/
  clean/
outputs/
  forecasts/
  metrics/
  plots/
```

Do not commit large generated data unless explicitly requested.

## Phase 0: Project Scaffold

Goal: create the minimal Python project shape.

Deliverables:

- `requirements.txt`
- `src/hourly_prediction/`
- `scripts/`
- `tests/`
- `.gitignore` for generated data, outputs, caches, virtual environments, and secrets.

Acceptance criteria:

- The repository has a clear package/script split.
- Generated data paths are ignored.
- No secrets or API keys are required for public candle fetching.

## Phase 1: Data Ingestion

Goal: fetch raw `BTC/USDT` candles for `1h` and `15m`.

Use:

```text
CCXT
exchange: binance
symbol: BTC/USDT
timeframes: 1h, 15m
```

Deliverables:

- `src/hourly_prediction/data.py`
- `scripts/fetch_ohlcv.py`
- Raw CSV outputs under `data/raw/`

Script behavior:

```text
python scripts/fetch_ohlcv.py --symbol BTC/USDT --timeframe 1h --limit 1000
python scripts/fetch_ohlcv.py --symbol BTC/USDT --timeframe 15m --limit 1000
```

Acceptance criteria:

- Fetches candles through CCXT with rate limiting enabled.
- Writes raw CSV with deterministic file names.
- Includes source metadata where practical: exchange, symbol, timeframe, fetch time.
- Does not use chart screenshots.

## Phase 2: Candle Validation

Goal: convert raw candles into clean, model-ready datasets.

Deliverables:

- `src/hourly_prediction/validation.py`
- `scripts/validate_ohlcv.py`
- Clean CSV outputs under `data/clean/`
- `tests/test_validation.py`

Validation checks:

- Required columns exist.
- Timestamps are UTC and sorted ascending.
- No duplicate timestamps.
- Expected interval spacing for `1h` and `15m`.
- No unfinished latest candle.
- `open`, `high`, `low`, `close`, `volume`, `amount` are finite.
- `amount` is present or computed as `close * volume`.

Acceptance criteria:

- Validation fails loudly on gaps and duplicates.
- Clean outputs contain only closed candles.
- Tests cover gaps, duplicates, bad ordering, and unfinished candles.

## Phase 2.5: One-Command Candle Refresh

Goal: provide a convenience wrapper for the completed fetch and validation flow.

Deliverables:

- `src/hourly_prediction/refresh.py`
- `scripts/refresh_candles.py`

Script behavior:

```text
python scripts/refresh_candles.py --symbol BTC/USDT --limit 1000
```

Acceptance criteria:

- Refreshes `1h` and `15m` by default.
- Saves raw CSVs under `data/raw/`.
- Saves clean CSVs under `data/clean/`.
- Prints the raw and clean output paths for each timeframe.
- Does not add Kronos inference, evaluation, backtesting, or new timeframes.

## Phase 3: Kronos Integration

Goal: run one next-candle forecast for each timeframe.

Deliverables:

- `src/hourly_prediction/kronos_runner.py`
- `scripts/run_kronos_forecast.py`
- Forecast CSVs under `outputs/forecasts/`

Model settings:

```text
model: Kronos-small or Kronos-base
lookback: 400-512
pred_len: 1
timeframes: 1h, 15m
```

Kronos input shape:

```text
x_df = open, high, low, close, volume, amount
x_timestamp = historical candle timestamps
y_timestamp = next expected timestamp only
```

Acceptance criteria:

- Forecast output includes timestamp, predicted OHLCV fields, model name, timeframe, lookback, and run time.
- `pred_len` is `1`.
- `1h` and `15m` are run as separate regular time-series forecasts.
- No future candles are available to the forecast function.

## Phase 4: Walk-Forward Evaluation

Goal: evaluate forecasts historically without future leakage.

Deliverables:

- `src/hourly_prediction/evaluation.py`
- `scripts/evaluate_forecast.py`
- Metrics under `outputs/metrics/`
- `tests/test_evaluation_windows.py`

Evaluation method:

For each valid historical decision timestamp:

```text
input window = previous lookback closed candles
target = next closed candle
forecast = Kronos prediction for that next candle
```

Baselines:

- Naive close persistence: next close equals current close.
- Optional moving average baseline.
- Optional random direction baseline.

Minimum metrics:

- MAE.
- RMSE.
- Directional accuracy.
- Predicted return vs actual return.

Acceptance criteria:

- Evaluation windows cannot include the target candle in the input.
- Metrics compare Kronos against at least the naive baseline.
- Results are saved as CSV or JSON.
- Tests prove there is no off-by-one leakage.

## Phase 5: Simple Paper Signal

Goal: translate forecasts into a non-live, inspectable paper signal.

This phase comes only after Phase 4.

Initial signal:

```text
1h forecast predicts direction
15m forecast confirms short-term direction
no trade if forecasts disagree
```

Deliverables:

- Backtest or signal script.
- Signal CSV.
- Fee/slippage assumptions documented in output metadata.

Acceptance criteria:

- No live order placement.
- Includes fees and slippage.
- Shows performance versus no-trade and naive baseline behavior.
- Clearly labels results as research/backtest output.

## LLM Agent Checklist

Before editing:

- Read `AGENTS.md`.
- Read `docs/kronos-data-ingestion-notes.md`.
- Read this roadmap.
- Confirm the active scope is `1h` and `15m`, `pred_len=1`.

During implementation:

- Make one phase or sub-phase change at a time.
- Add focused tests for new logic.
- Save outputs to predictable paths.
- Keep functions small and data contracts explicit.
- Avoid hidden global state and notebook-only workflows.

Before final response:

- Report changed files.
- Report commands run.
- Report any tests not run.
- State any remaining risks or assumptions.

## Prompt Pattern For Future LLM Work

Use prompts like this:

```text
Implement Phase 1 from docs/llm-implementation-roadmap.md.
Stay within the current scope: BTC/USDT, 1h and 15m, pred_len=1.
Add or update tests where relevant.
Do not add Kronos inference yet.
Save generated data under data/raw and do not commit large outputs.
```

For later phases:

```text
Implement Phase 2 only.
Do not modify the data-fetching behavior except where validation needs a stable interface.
Add tests for timestamp ordering, duplicates, gaps, and unfinished candles.
```

## Out Of Scope For Now

- Live trading.
- API key management beyond public market data.
- 24h prediction horizons.
- `1m` candles.
- `4h` regime filtering.
- Multi-asset portfolios.
- Portfolio optimization.
- News, sentiment, or chart screenshot extraction.
