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

## 2026-06-06 Phase 4D Model and Sampling Comparison

Input comparison:

```text
outputs/metrics/binance_BTCUSDT_20260606T205214Z_phase4d_walk_forward_model_comparison.csv
```

Run shape:

- Symbol: `BTC/USDT`
- Timeframes: `1h`, `15m`
- Lookback: `512`
- Prediction length: `1`
- Windows: `100` recent walk-forward targets per timeframe
- SMA window: `20`
- Sampling: `top_p=0.9`

Summary:

| Model | Sample Count | Timeframe | Rows | Kronos MAE | Naive MAE | SMA MAE | Kronos Directional Accuracy | Random Direction Accuracy | Beats Naive MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NeoQuasar/Kronos-base` | 1 | `15m` | 100 | 251.595145 | 157.150900 | 331.751630 | 0.50 | 0.27 | no |
| `NeoQuasar/Kronos-base` | 1 | `1h` | 100 | 531.015483 | 403.730900 | 951.203450 | 0.58 | 0.28 | no |
| `NeoQuasar/Kronos-small` | 1 | `15m` | 100 | 239.782008 | 157.150900 | 331.751630 | 0.43 | 0.27 | no |
| `NeoQuasar/Kronos-small` | 1 | `1h` | 100 | 640.973461 | 403.730900 | 951.203450 | 0.46 | 0.28 | no |
| `NeoQuasar/Kronos-small` | 3 | `15m` | 100 | 229.470303 | 157.150900 | 331.751630 | 0.44 | 0.27 | no |
| `NeoQuasar/Kronos-small` | 3 | `1h` | 100 | 530.735658 | 403.730900 | 951.203450 | 0.58 | 0.28 | no |

The best tested Kronos configuration by MAE was `Kronos-small` with
`sample_count=3` on both timeframes, but it still did not beat naive close
persistence. All tested Kronos configurations beat the 20-candle SMA baseline
on MAE. Phase 5 paper-signal work should remain blocked; the next research
step should either broaden evaluation history or investigate why naive close
persistence remains stronger on close-error metrics.

## 2026-06-06 Phase 4E Broadened Even-Window Coverage

Input comparison:

```text
outputs/metrics/binance_BTCUSDT_20260606T205214Z_phase4e_even_walk_forward_model_comparison.csv
```

Run shape:

- Symbol: `BTC/USDT`
- Timeframes: `1h`, `15m`
- Lookback: `512`
- Prediction length: `1`
- Windows: `400` evenly spaced walk-forward targets per timeframe
- SMA window: `20`
- Sampling: `top_p=0.9`

Summary:

| Model | Sample Count | Timeframe | Rows | Kronos MAE | Naive MAE | SMA MAE | Kronos Directional Accuracy | Random Direction Accuracy | Beats Naive MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NeoQuasar/Kronos-base` | 1 | `15m` | 400 | 273.606982 | 192.337150 | 457.178834 | 0.5275 | 0.3275 | no |
| `NeoQuasar/Kronos-base` | 1 | `1h` | 400 | 328.870392 | 224.958375 | 599.485101 | 0.5200 | 0.3100 | no |
| `NeoQuasar/Kronos-small` | 3 | `15m` | 400 | 246.769309 | 192.337150 | 457.178834 | 0.5450 | 0.3275 | no |
| `NeoQuasar/Kronos-small` | 3 | `1h` | 400 | 282.469495 | 224.958375 | 599.485101 | 0.4925 | 0.3100 | no |

The broader even-window run confirms the Phase 4D result across more historical
targets from the current clean datasets: neither tested Kronos configuration
beats naive close persistence on MAE/RMSE for `1h` or `15m`. Both tested
configurations remain better than the 20-candle SMA baseline on MAE. Phase 5
paper-signal work should remain blocked; the next research slice should focus
on data depth, target formulation, or additional non-trading diagnostics.

## 2026-06-07 Phase 4F Target and Regime Diagnostics

Input diagnostics:

```text
outputs/metrics/binance_BTCUSDT_20260606T205214Z_phase4f_even_walk_forward_regime_diagnostics.csv
```

Run shape:

- Inputs: Phase 4E `even` metrics for `Kronos-small sample_count=3` and `Kronos-base sample_count=1`
- Regimes: 3 deterministic quantile buckets by absolute actual return per timeframe
- Rows: 12 aggregate rows, covering 2 configs x 2 timeframes x 3 regimes

Summary:

| Model | Sample Count | Timeframe | Regime | Rows | Avg Abs Actual Return | Kronos MAE | Naive MAE | Kronos / Naive MAE Ratio | Directional Accuracy | Forecast/Actual Return Corr |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NeoQuasar/Kronos-base` | 1 | `15m` | `q1_of_3_abs_return` | 133 | 0.000707 | 199.745772 | 46.273534 | 4.316631 | 0.548872 | 0.076418 |
| `NeoQuasar/Kronos-base` | 1 | `15m` | `q2_of_3_abs_return` | 133 | 0.002378 | 228.468739 | 153.712857 | 1.486335 | 0.503759 | 0.002437 |
| `NeoQuasar/Kronos-base` | 1 | `15m` | `q3_of_3_abs_return` | 134 | 0.005882 | 391.718378 | 375.646791 | 1.042784 | 0.529851 | 0.024672 |
| `NeoQuasar/Kronos-base` | 1 | `1h` | `q1_of_3_abs_return` | 133 | 0.000558 | 192.535101 | 41.404286 | 4.650125 | 0.511278 | 0.076235 |
| `NeoQuasar/Kronos-base` | 1 | `1h` | `q2_of_3_abs_return` | 133 | 0.002082 | 224.786544 | 154.776165 | 1.452333 | 0.511278 | 0.108928 |
| `NeoQuasar/Kronos-base` | 1 | `1h` | `q3_of_3_abs_return` | 134 | 0.006914 | 567.495359 | 476.801119 | 1.190214 | 0.537313 | 0.139909 |
| `NeoQuasar/Kronos-small` | 3 | `15m` | `q1_of_3_abs_return` | 133 | 0.000707 | 138.219094 | 46.273534 | 2.987001 | 0.578947 | 0.192799 |
| `NeoQuasar/Kronos-small` | 3 | `15m` | `q2_of_3_abs_return` | 133 | 0.002378 | 197.566624 | 153.712857 | 1.285297 | 0.563910 | 0.028737 |
| `NeoQuasar/Kronos-small` | 3 | `15m` | `q3_of_3_abs_return` | 134 | 0.005882 | 403.344948 | 375.646791 | 1.073735 | 0.492537 | -0.107864 |
| `NeoQuasar/Kronos-small` | 3 | `1h` | `q1_of_3_abs_return` | 133 | 0.000558 | 141.347768 | 41.404286 | 3.413844 | 0.488722 | -0.056514 |
| `NeoQuasar/Kronos-small` | 3 | `1h` | `q2_of_3_abs_return` | 133 | 0.002082 | 200.101336 | 154.776165 | 1.292843 | 0.556391 | 0.188036 |
| `NeoQuasar/Kronos-small` | 3 | `1h` | `q3_of_3_abs_return` | 134 | 0.006914 | 504.291546 | 476.801119 | 1.057656 | 0.432836 | 0.012529 |

Kronos does not only fail in low-move/noisy regimes: neither tested config
beats naive MAE in any return bucket. The gap is largest in tiny-move regimes,
where naive close persistence is very hard to beat, and narrows materially in
the largest-return bucket. Forecasted-vs-actual return correlations are weak
across buckets, so Phase 5 remains blocked. The next research slice should
focus on data depth, target construction, or model input formulation rather
than signal generation.

## 2026-06-07 Phase 4H Historical Walk-Forward Evaluation

Input manifest:

```text
outputs/manifests/binancepublic_BTCUSDT_20260607T071726Z_manifest.json
```

Input comparison:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4h_even_walk_forward_model_comparison.csv
```

Input regimes:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4h_even_walk_forward_regime_diagnostics.csv
```

Run shape:

- Source: Binance public monthly spot klines, `2025-01` through `2026-05`
- Symbol: `BTC/USDT`
- Timeframes: `1h`, `15m`
- Lookback: `512`
- Prediction length: `1`
- Windows: `1000` evenly spaced walk-forward targets per timeframe
- SMA window: `20`
- Sampling: `top_p=0.9`

Summary:

| Model | Sample Count | Timeframe | Rows | Kronos MAE | Naive MAE | SMA MAE | Kronos RMSE | Naive RMSE | Directional Accuracy | Random Direction Accuracy | Beats Naive MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NeoQuasar/Kronos-base` | 1 | `15m` | 1000 | 198.006689 | 149.233380 | 379.131199 | 323.690120 | 238.781881 | 0.516 | 0.343 | no |
| `NeoQuasar/Kronos-base` | 1 | `1h` | 1000 | 397.099599 | 274.777730 | 790.347289 | 566.596991 | 402.717696 | 0.503 | 0.318 | no |
| `NeoQuasar/Kronos-small` | 3 | `15m` | 1000 | 178.774395 | 149.233380 | 379.131199 | 300.691439 | 238.781881 | 0.511 | 0.343 | no |
| `NeoQuasar/Kronos-small` | 3 | `1h` | 1000 | 345.306029 | 274.777730 | 790.347289 | 491.713236 | 402.717696 | 0.519 | 0.318 | no |

Regime diagnostics:

| Model | Sample Count | Timeframe | Regime | Rows | Kronos / Naive MAE Ratio | Directional Accuracy | Forecast/Actual Return Corr |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `NeoQuasar/Kronos-base` | 1 | `15m` | `q1_of_3_abs_return` | 333 | 3.411471 | 0.528529 | 0.094496 |
| `NeoQuasar/Kronos-base` | 1 | `15m` | `q2_of_3_abs_return` | 333 | 1.420170 | 0.513514 | 0.012793 |
| `NeoQuasar/Kronos-base` | 1 | `15m` | `q3_of_3_abs_return` | 334 | 1.094302 | 0.505988 | 0.227848 |
| `NeoQuasar/Kronos-base` | 1 | `1h` | `q1_of_3_abs_return` | 333 | 4.373915 | 0.495495 | -0.002678 |
| `NeoQuasar/Kronos-base` | 1 | `1h` | `q2_of_3_abs_return` | 333 | 1.696442 | 0.477477 | -0.021014 |
| `NeoQuasar/Kronos-base` | 1 | `1h` | `q3_of_3_abs_return` | 334 | 1.096971 | 0.535928 | -0.023089 |
| `NeoQuasar/Kronos-small` | 3 | `15m` | `q1_of_3_abs_return` | 333 | 2.828761 | 0.510511 | 0.102555 |
| `NeoQuasar/Kronos-small` | 3 | `15m` | `q2_of_3_abs_return` | 333 | 1.207331 | 0.522523 | 0.049224 |
| `NeoQuasar/Kronos-small` | 3 | `15m` | `q3_of_3_abs_return` | 334 | 1.036348 | 0.500000 | 0.235967 |
| `NeoQuasar/Kronos-small` | 3 | `1h` | `q1_of_3_abs_return` | 333 | 3.185471 | 0.543544 | 0.035249 |
| `NeoQuasar/Kronos-small` | 3 | `1h` | `q2_of_3_abs_return` | 333 | 1.324878 | 0.513514 | -0.020366 |
| `NeoQuasar/Kronos-small` | 3 | `1h` | `q3_of_3_abs_return` | 334 | 1.060047 | 0.500000 | 0.004924 |

The deeper historical run confirms the prior gate. Neither tested Kronos
configuration beats naive close persistence on MAE/RMSE for either timeframe,
and Kronos does not beat naive in any absolute-return regime bucket. The
Kronos/naive gap is still largest in tiny-return regimes and narrowest in the
largest-return bucket, but it remains above `1.0`. Phase 5 remains blocked;
the next research slice should investigate target or input formulation rather
than increasing walk-forward scale again.

## 2026-06-07 Phase 4I Target Formulation Diagnostics

Input diagnostics:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4i_target_formulation_diagnostics.csv
```

Run shape:

- Inputs: Phase 4H historical `even` metrics for `Kronos-small sample_count=3` and `Kronos-base sample_count=1`
- Thresholded direction cutoffs: `0`, `5`, `10`, and `25` bps by absolute actual return
- Rows: 4 aggregate rows, covering 2 configs x 2 timeframes

Summary:

| Model | Sample Count | Timeframe | Rows | Kronos Close MAE | Naive Close MAE | Kronos Return MAE | Naive Return MAE | Forecast/Actual Return Corr | 10 bps Direction Accuracy | 25 bps Direction Accuracy | Beats Naive Close MAE | Beats Naive Return MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `NeoQuasar/Kronos-base` | 1 | `15m` | 1000 | 198.006689 | 149.233380 | 0.002148 | 0.001612 | 0.198888 | 0.507692 | 0.553763 | no | no |
| `NeoQuasar/Kronos-base` | 1 | `1h` | 1000 | 397.099599 | 274.777730 | 0.004344 | 0.003017 | -0.019652 | 0.512640 | 0.530660 | no | no |
| `NeoQuasar/Kronos-small` | 3 | `15m` | 1000 | 178.774395 | 149.233380 | 0.001932 | 0.001612 | 0.214861 | 0.509615 | 0.510753 | no | no |
| `NeoQuasar/Kronos-small` | 3 | `1h` | 1000 | 345.306029 | 274.777730 | 0.003787 | 0.003017 | 0.001925 | 0.509831 | 0.500000 | no | no |

The target-formulation diagnostics show that the issue is not only raw close
calibration: Kronos also loses to naive on return-error MAE for both tested
models and timeframes. Thresholded directional accuracy is only slightly above
coin-flip levels and does not provide enough evidence to unblock Phase 5. The
next research slice should change the input or target formulation, for example
evaluating return-space or normalized-price targets, before any signal or
backtest work.

## 2026-06-07 Phase 4J Leakage-Safe Forecast Calibration

Input diagnostics:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4j_forecast_calibration.csv
```

Run shape:

- Inputs: Phase 4H historical `even` metrics for `Kronos-small sample_count=3` and `Kronos-base sample_count=1`
- Split: chronological `70%` train, `30%` test per model/config/timeframe
- Calibration methods: uncalibrated Kronos return, train-only bias correction, train-only linear return calibration
- Thresholded direction cutoffs: `0`, `5`, `10`, and `25` bps by absolute actual return
- Rows: 4 aggregate rows, covering 2 configs x 2 timeframes

Summary:

| Model | Sample Count | Timeframe | Train Rows | Test Rows | Uncal Return MAE | Bias Return MAE | Linear Return MAE | Naive Return MAE | Uncal Close MAE | Bias Close MAE | Linear Close MAE | Naive Close MAE | Linear Beats Naive Close MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NeoQuasar/Kronos-base` | 1 | `15m` | 700 | 300 | 0.002325 | 0.002330 | 0.001711 | 0.001647 | 173.875528 | 174.319891 | 127.737027 | 122.727800 | no |
| `NeoQuasar/Kronos-base` | 1 | `1h` | 700 | 300 | 0.004873 | 0.004867 | 0.003386 | 0.003392 | 363.867726 | 363.376804 | 252.860384 | 253.248533 | yes |
| `NeoQuasar/Kronos-small` | 3 | `15m` | 700 | 300 | 0.001958 | 0.001965 | 0.001661 | 0.001647 | 145.687210 | 146.302132 | 123.770402 | 122.727800 | no |
| `NeoQuasar/Kronos-small` | 3 | `1h` | 700 | 300 | 0.004214 | 0.004200 | 0.003392 | 0.003392 | 314.331031 | 313.397273 | 253.254440 | 253.248533 | no |

The leakage-safe calibration check does not provide enough evidence to unblock
Phase 5. Linear calibration produces one tiny out-of-sample edge on `Kronos-base`
`1h`, beating naive close MAE by about `0.39` USD and return MAE by about
`0.000006`, but the other three model/timeframe rows still lose to naive. Test
set forecast-vs-actual return correlations are weak or negative across rows, so
the next research slice should reformulate the model input or target directly
rather than treat calibration as a trading-ready signal.

## 2026-06-07 Phase 4K Input Normalization Experiment

Input comparison:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4k_input_transform_model_comparison.csv
```

Related diagnostics:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4k_input_transform_regime_diagnostics.csv
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4k_input_transform_target_formulation.csv
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4k_input_transform_forecast_calibration.csv
```

Run shape:

- Inputs: Binance public historical manifest from Phase 4H
- Models: `Kronos-small sample_count=3` and `Kronos-base sample_count=1`
- Input transforms: `raw` and `relative`
- Windows: `500` evenly spaced targets per timeframe, `1000` rows per run
- Timeframes: `1h`, `15m`

Summary:

| Model | Sample Count | Input Transform | Timeframe | Rows | Kronos MAE | Naive MAE | Kronos RMSE | Naive RMSE | Directional Accuracy | Beats Naive MAE |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NeoQuasar/Kronos-base` | 1 | `raw` | `15m` | 500 | 187.949908 | 137.776240 | 283.785438 | 221.836705 | 0.510 | no |
| `NeoQuasar/Kronos-base` | 1 | `relative` | `15m` | 500 | 193.894492 | 137.776240 | 289.375097 | 221.836705 | 0.522 | no |
| `NeoQuasar/Kronos-base` | 1 | `raw` | `1h` | 500 | 390.918133 | 279.443480 | 578.379409 | 423.381350 | 0.488 | no |
| `NeoQuasar/Kronos-base` | 1 | `relative` | `1h` | 500 | 382.862431 | 279.443480 | 538.019441 | 423.381350 | 0.536 | no |
| `NeoQuasar/Kronos-small` | 3 | `raw` | `15m` | 500 | 164.631058 | 137.776240 | 244.046875 | 221.836705 | 0.504 | no |
| `NeoQuasar/Kronos-small` | 3 | `relative` | `15m` | 500 | 167.522711 | 137.776240 | 251.651290 | 221.836705 | 0.498 | no |
| `NeoQuasar/Kronos-small` | 3 | `raw` | `1h` | 500 | 317.723873 | 279.443480 | 454.806388 | 423.381350 | 0.532 | no |
| `NeoQuasar/Kronos-small` | 3 | `relative` | `1h` | 500 | 342.270729 | 279.443480 | 504.897857 | 423.381350 | 0.526 | no |

Relative input normalization does not beat naive close persistence on direct
walk-forward MAE/RMSE for any model/timeframe row. It improves `Kronos-base`
`1h` versus raw input and raises its directional accuracy, but it worsens both
`15m` rows and `Kronos-small 1h`. The calibration report again shows only a
tiny, isolated linear-calibrated close-MAE edge for `Kronos-base 1h relative`
on the held-out split; it does not beat naive on return MAE. Phase 5 remains
blocked. The next research slice should focus on a different target definition
or a more explicit return-space forecasting experiment rather than treating
input normalization as sufficient.

## 2026-06-07 Phase 4L Return-Space Target Experiment

Input comparison:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4l_return_space_model_comparison.csv
```

Related diagnostics:

```text
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4l_return_space_regime_diagnostics.csv
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4l_return_space_target_formulation.csv
outputs/metrics/binancepublic_BTCUSDT_20260607T071726Z_phase4l_return_space_forecast_calibration.csv
```

Run shape:

- Inputs: Binance public historical manifest from Phase 4H
- Models: `Kronos-small sample_count=3` and `Kronos-base sample_count=1`
- Input transforms: `raw`, `relative`, and `log-return`
- Windows: `500` evenly spaced targets per timeframe, `1000` rows per transform/model run
- Timeframes: `1h`, `15m`

Summary:

| Model | Sample Count | Input Transform | Timeframe | Rows | Kronos MAE | Naive MAE | Kronos RMSE | Naive RMSE | Directional Accuracy | Beats Naive MAE |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `NeoQuasar/Kronos-base` | 1 | `log-return` | `15m` | 500 | 172.348211 | 137.776240 | 243.597339 | 221.836705 | 0.508 | no |
| `NeoQuasar/Kronos-base` | 1 | `raw` | `15m` | 500 | 187.949908 | 137.776240 | 283.785438 | 221.836705 | 0.510 | no |
| `NeoQuasar/Kronos-base` | 1 | `relative` | `15m` | 500 | 193.894492 | 137.776240 | 289.375097 | 221.836705 | 0.522 | no |
| `NeoQuasar/Kronos-base` | 1 | `log-return` | `1h` | 500 | 365.519396 | 279.443480 | 514.199547 | 423.381350 | 0.470 | no |
| `NeoQuasar/Kronos-base` | 1 | `raw` | `1h` | 500 | 390.918133 | 279.443480 | 578.379409 | 423.381350 | 0.488 | no |
| `NeoQuasar/Kronos-base` | 1 | `relative` | `1h` | 500 | 382.862431 | 279.443480 | 538.019441 | 423.381350 | 0.536 | no |
| `NeoQuasar/Kronos-small` | 3 | `log-return` | `15m` | 500 | 163.114536 | 137.776240 | 238.475103 | 221.836705 | 0.500 | no |
| `NeoQuasar/Kronos-small` | 3 | `raw` | `15m` | 500 | 164.631058 | 137.776240 | 244.046875 | 221.836705 | 0.504 | no |
| `NeoQuasar/Kronos-small` | 3 | `relative` | `15m` | 500 | 167.522711 | 137.776240 | 251.651290 | 221.836705 | 0.498 | no |
| `NeoQuasar/Kronos-small` | 3 | `log-return` | `1h` | 500 | 334.284402 | 279.443480 | 491.195919 | 423.381350 | 0.482 | no |
| `NeoQuasar/Kronos-small` | 3 | `raw` | `1h` | 500 | 317.723873 | 279.443480 | 454.806388 | 423.381350 | 0.532 | no |
| `NeoQuasar/Kronos-small` | 3 | `relative` | `1h` | 500 | 342.270729 | 279.443480 | 504.897857 | 423.381350 | 0.526 | no |

The log-return transform improves direct close-error MAE/RMSE versus raw and
relative for `Kronos-base 15m`, `Kronos-base 1h`, and `Kronos-small 15m`, but
it still does not beat naive close persistence on any direct walk-forward row.
For `Kronos-small 1h`, raw input remains better than log-return. The calibration
report shows small held-out linear-calibration edges over naive for both `1h`
log-return rows, but not for either `15m` row and not on uncalibrated metrics.
Phase 5 remains blocked. The next research slice should investigate whether a
different model objective or explicit post-model return calibration can be
validated robustly across more than the `1h` held-out split.
