# Walk-Forward Evaluation Notes

## 2026-06-06 Bounded Kronos-small Smoke

Input metrics:

```text
outputs/metrics/binance_BTCUSDT_20260606T095247Z_kronos-small_walk_forward_metrics.csv
```

Run shape:

- Symbol: `BTC/USDT`
- Timeframes: `1h`, `15m`
- Model: `NeoQuasar/Kronos-small`
- Lookback: `512`
- Prediction length: `1`
- Windows: `20` recent walk-forward targets per timeframe

Summary:

| Timeframe | Rows | Kronos MAE | Naive MAE | Kronos Directional Accuracy | Naive Directional Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `15m` | 20 | 269.031227 | 186.1170 | 0.45 | 0.00 |
| `1h` | 20 | 718.050625 | 525.0215 | 0.60 | 0.00 |

The first bounded walk-forward run did not beat naive close persistence on close
error for either timeframe. Treat this as a Phase 4 review gate, not a trading
signal. Phase 5 paper-signal work should wait until the evaluation results are
reviewed and, if needed, rerun with a larger sample or additional baselines.

## Phase 4B Baseline Hardening

The next evaluation slice adds a simple moving-average close baseline to every
walk-forward row.

- Default SMA window: `20` input candles.
- SMA prediction: next close equals the mean of the last `20` input closes.
- The SMA baseline must use only the input window and never the target candle.
- Summary reports should compare Kronos, naive close persistence, and SMA.

After this change, rerun walk-forward evaluation with a larger bounded sample,
for example `--max-windows 100`, before considering Phase 5 paper-signal work.

## 2026-06-06 Phase 4B 100-Window SMA Comparison

Input metrics:

```text
outputs/metrics/binance_BTCUSDT_20260606T143156Z_kronos-small_walk_forward_metrics.csv
```

Summary:

| Timeframe | Rows | Kronos MAE | Naive MAE | SMA MAE | Kronos Directional Accuracy | Naive Directional Accuracy | SMA Directional Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15m` | 100 | 345.129412 | 209.009200 | 517.067285 | 0.39 | 0.00 | 0.56 |
| `1h` | 100 | 589.100241 | 426.239700 | 1054.148970 | 0.55 | 0.00 | 0.49 |

On this larger bounded sample, Kronos still did not beat naive close
persistence on MAE/RMSE for either timeframe. SMA had better directional
accuracy on `15m`, but materially worse close-error metrics. Continue treating
these as research evaluation artifacts, not paper-trading signals.
