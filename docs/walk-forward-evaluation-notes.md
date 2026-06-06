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
