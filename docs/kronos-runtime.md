# Kronos CUDA Runtime

This project uses the base `requirements.txt` for ingestion and validation. Kronos
model inference is kept in a separate setup because it depends on PyTorch, CUDA,
and a local checkout of the Kronos repository.

## Target Runtime

- Python: `3.12`
- GPU: NVIDIA RTX 3060 12GB
- Device: `cuda:0`
- PyTorch wheel: CUDA `13.0`
- First model: `NeoQuasar/Kronos-small`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`
- Max context: `512`

The current shell Python may not be suitable. The runtime checker intentionally
fails unless it is run from Python `3.12`.

## Setup

Create and activate a Python `3.12` virtual environment, then install the base
project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the PyTorch CUDA wheel first:

```powershell
python -m pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Install the remaining Kronos dependencies:

```powershell
python -m pip install -r requirements-kronos.txt
```

Clone Kronos locally into the ignored vendor path:

```powershell
New-Item -ItemType Directory -Force vendor
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
```

Do not commit `vendor/Kronos/`.

## Checks

CUDA and import preflight:

```powershell
python scripts/check_kronos_runtime.py --kronos-repo-path vendor/Kronos --device cuda:0 --mode cuda
```

Model-load smoke test:

```powershell
python scripts/check_kronos_runtime.py --kronos-repo-path vendor/Kronos --device cuda:0 --mode model
```

The check must not silently fall back to CPU. If CUDA is unavailable, Python is
not `3.12`, the Kronos repo is missing, or the required Kronos classes cannot be
imported, the command exits non-zero.

## Out Of Scope

This runtime spike does not create forecasts, evaluation metrics, backtests,
trading signals, new timeframes, or 24-hour horizons.
