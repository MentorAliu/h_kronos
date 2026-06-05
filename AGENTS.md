# AGENTS.md

## Project

This repository is for building a reproducible Kronos-based financial time-series forecasting pipeline.

The first working target is:

```text
BTC/USDT 1h and 15m candles -> validated OHLCV datasets -> next-candle Kronos forecasts -> evaluation against baselines
```

Treat this as a research and evaluation pipeline first, not a live trading bot.

## Initial Scope

Use the smallest reliable slice before adding complexity:

- Exchange/data source: Binance through CCXT, with Binance public data for larger historical backfills.
- Symbol: `BTC/USDT`.
- Primary timeframes: `1h` and `15m`.
- Lookback: `400-512` closed candles.
- Prediction length: `1` candle for each timeframe.
- Model: open Kronos checkpoint, starting with `Kronos-small` or `Kronos-base`.

Initial forecast targets:

- `1h` model predicts the next closed `1h` candle.
- `15m` model predicts the next closed `15m` candle.

Do not add `1m`, `4h`, or 24-hour forecast horizons unless there is a specific experiment proving the need.

## Architecture

Keep pipeline stages separate:

```text
fetch -> validate -> forecast -> evaluate -> backtest
```

Expected initial files:

```text
requirements.txt
scripts/fetch_ohlcv.py
scripts/validate_ohlcv.py
scripts/run_kronos_forecast.py
scripts/evaluate_forecast.py
data/
outputs/
docs/
```

Do not combine ingestion, model inference, and evaluation into one large script.

## Data Rules

Use raw exchange OHLCV whenever available. Do not use chart screenshots as a data source unless raw candles are unavailable.

Kronos input data should include:

```text
open, high, low, close, volume, amount
```

Where:

```text
amount = close * volume
```

Keep timestamps aligned to the candle rows and use only closed candles for any forecast or backtest decision.

## Leakage Prevention

Avoid future leakage rigorously:

- Never use an unfinished live candle as historical input.
- In walk-forward evaluation, build each forecast using only data available at that historical timestamp.
- Do not normalize using future data.
- Do not let target-period values influence feature construction.
- Keep `x_timestamp` and `y_timestamp` alignment explicit and testable.

This is the highest-risk failure mode in the project.

## Artifact Rules

Every meaningful run should save inspectable outputs:

- Raw candles.
- Cleaned candles.
- Forecast CSV.
- Metrics JSON or CSV.
- Optional plots.

Avoid workflows where important results only appear in terminal output.

Use local data/output folders, and avoid committing large generated datasets unless explicitly requested.

## Evaluation

Kronos must be compared against simple baselines before any trading interpretation:

- Naive close persistence: future close equals current close.
- Simple moving average baseline.
- Random direction baseline where useful.

Minimum metrics:

- MAE.
- RMSE.
- Directional accuracy.
- Forecasted return vs actual return.

For any backtest, include fees and slippage assumptions.

## LLM Agent Workflow

Agents should work in small, verifiable slices:

1. Read existing docs and scripts before editing.
2. Make focused changes with clear inputs and outputs.
3. Add tests for data validation and timestamp alignment.
4. Run the narrowest useful verification command.
5. Record assumptions in docs when they affect experiments.

Use LLM assistance for:

- Scaffolding scripts.
- Writing tests.
- Diagnosing data or model failures.
- Reviewing leakage risks.
- Summarizing metrics and experiment results.

Do not use an LLM as the source of truth for:

- Market prices.
- Profitability claims.
- Live trading decisions.

## Documentation

Keep project decisions in `docs/`.

Current seed note:

```text
docs/kronos-data-ingestion-notes.md
docs/llm-implementation-roadmap.md
```

Update documentation when changing:

- Data source.
- Symbol universe.
- Timeframe plan.
- Lookback or prediction length.
- Model checkpoint.
- Evaluation metrics.
- Backtest assumptions.

## Safety

No live trading integration should be added until:

- Data ingestion is stable.
- Walk-forward evaluation exists.
- A baseline comparison exists.
- Backtest assumptions are explicit.
- Results are reviewed out of sample.

Any API key handling must use environment variables or local ignored config files. Never commit secrets.
