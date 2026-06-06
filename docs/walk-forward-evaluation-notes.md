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

## 2026-06-06 Phase 4C Forecast Diagnostics

Input diagnostics:

```text
outputs/metrics/binance_BTCUSDT_20260606T143156Z_kronos-small_walk_forward_diagnostics.csv
```

Summary:

| Timeframe | Rows | Kronos Mean Signed Error | Naive Mean Signed Error | SMA Mean Signed Error | Kronos / Naive MAE Ratio | Kronos / SMA MAE Ratio | Random Direction Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15m` | 100 | 67.034759 | 12.280000 | 149.040575 | 1.651264 | 0.667475 | 0.24 |
| `1h` | 100 | 115.952725 | 86.248100 | 1019.555880 | 1.382087 | 0.558840 | 0.47 |

Kronos shows a positive signed close-error bias on both timeframes. It remains
worse than naive close persistence on MAE, but better than the 20-candle SMA
baseline on MAE. Directional behavior needs review separately from close-error
metrics; this remains a diagnostics artifact and not a signal/backtest result.

## 2026-06-06 Phase 4B/4C Main Recreated Run

Input metrics:

```text
outputs/metrics/binance_BTCUSDT_20260606T205214Z_kronos-small_walk_forward_metrics.csv
```

Derived artifacts:

```text
outputs/metrics/binance_BTCUSDT_20260606T205214Z_kronos-small_walk_forward_summary.csv
outputs/metrics/binance_BTCUSDT_20260606T205214Z_kronos-small_walk_forward_diagnostics.csv
```

Summary:

| Timeframe | Rows | Kronos MAE | Naive MAE | SMA MAE | Kronos Directional Accuracy | Naive Directional Accuracy | SMA Directional Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15m` | 100 | 242.854920 | 157.150900 | 331.751630 | 0.48 | 0.00 | 0.59 |
| `1h` | 100 | 638.611161 | 403.730900 | 951.203450 | 0.54 | 0.00 | 0.49 |

Diagnostics:

| Timeframe | Rows | Kronos Mean Signed Error | Naive Mean Signed Error | SMA Mean Signed Error | Kronos / Naive MAE Ratio | Kronos / SMA MAE Ratio | Random Direction Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15m` | 100 | -40.687661 | -9.822100 | -38.250200 | 1.545361 | 0.732038 | 0.27 |
| `1h` | 100 | 136.536733 | 67.042700 | 916.610360 | 1.581774 | 0.671372 | 0.28 |

The recreated main run confirms the same Phase 4 gate: Kronos-small is still
worse than naive close persistence on close-error metrics for both timeframes,
while beating the 20-candle SMA baseline on MAE. Phase 5 paper-signal work
should remain blocked until this baseline result is reviewed or a model/config
comparison improves the close-error profile.
