from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_KRONOS_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MAX_CONTEXT = 512
REQUIRED_KRONOS_CLASSES = ("Kronos", "KronosTokenizer", "KronosPredictor")
REQUIRED_PYTHON = (3, 12)


class KronosRuntimeError(RuntimeError):
    """Raised when the local Kronos CUDA runtime is not ready."""


def check_kronos_runtime(
    *,
    kronos_repo_path: Path,
    device: str,
    mode: str,
    torch_module: Any | None = None,
    python_version: tuple[int, ...] | None = None,
    model_name: str = DEFAULT_KRONOS_MODEL,
    tokenizer_name: str = DEFAULT_KRONOS_TOKENIZER,
    max_context: int = DEFAULT_MAX_CONTEXT,
) -> dict[str, Any]:
    """Validate the CUDA-first Kronos runtime and optionally load model weights."""

    version = _normalize_python_version(python_version)
    _check_python_version(version)
    torch = torch_module or _import_torch()
    cuda_report = _check_cuda(torch=torch, device=device)
    model_module = _import_kronos_model(kronos_repo_path)
    class_report = _check_required_classes(model_module)

    report: dict[str, Any] = {
        "python": ".".join(str(part) for part in version[:3]),
        **cuda_report,
        **class_report,
        "kronos_repo_path": str(kronos_repo_path),
        "mode": mode,
    }

    if mode == "cuda":
        return report
    if mode == "model":
        report.update(
            _load_kronos_model(
                model_module=model_module,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                device=device,
                max_context=max_context,
            )
        )
        return report

    raise KronosRuntimeError("Unsupported mode; expected 'cuda' or 'model'")


def print_runtime_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def _check_python_version(version: tuple[int, ...]) -> None:
    if version[:2] != REQUIRED_PYTHON:
        raise KronosRuntimeError(
            "Python 3.12 is required for the Kronos CUDA runtime. "
            f"Current Python is {'.'.join(str(part) for part in version[:3])}. "
            "Create and activate a Python 3.12 virtual environment before installing PyTorch."
        )


def _check_cuda(*, torch: Any, device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        raise KronosRuntimeError("CUDA is required; pass a device like 'cuda:0'.")

    if not torch.cuda.is_available():
        raise KronosRuntimeError("CUDA is required, but torch.cuda.is_available() is false.")

    device_index = _parse_cuda_device_index(device)
    device_name = torch.cuda.get_device_name(device_index)
    free_gb: float | None = None
    total_gb: float | None = None
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        free_gb = round(free_bytes / 1024**3, 2)
        total_gb = round(total_bytes / 1024**3, 2)

    return {
        "cuda_available": True,
        "device": device,
        "device_name": device_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
    }


def _import_kronos_model(kronos_repo_path: Path) -> ModuleType:
    repo_path = kronos_repo_path.resolve()
    if not repo_path.exists():
        raise KronosRuntimeError(f"Kronos repo path does not exist: {repo_path}")
    if not (repo_path / "model").exists():
        raise KronosRuntimeError(f"Kronos model package not found under: {repo_path}")

    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "model" or name.startswith("model.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(repo_path))
    try:
        model_module = importlib.import_module("model")
        if _has_required_classes(model_module):
            return model_module
        try:
            kronos_module = importlib.import_module("model.kronos")
        except ModuleNotFoundError as exc:
            if exc.name == "model.kronos":
                return model_module
            raise
        if _has_required_classes(kronos_module):
            return kronos_module
        return model_module
    except Exception as exc:  # pragma: no cover - preserves import failure details.
        raise KronosRuntimeError(f"Failed to import Kronos model package: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def _check_required_classes(model_module: ModuleType) -> dict[str, Any]:
    missing = [
        class_name
        for class_name in REQUIRED_KRONOS_CLASSES
        if not hasattr(model_module, class_name)
    ]
    if missing:
        raise KronosRuntimeError(
            "Kronos model package is missing required class(es): "
            + ", ".join(missing)
        )
    return {"kronos_classes": list(REQUIRED_KRONOS_CLASSES)}


def _has_required_classes(model_module: ModuleType) -> bool:
    return all(hasattr(model_module, class_name) for class_name in REQUIRED_KRONOS_CLASSES)


def _load_kronos_model(
    *,
    model_module: ModuleType,
    model_name: str,
    tokenizer_name: str,
    device: str,
    max_context: int,
) -> dict[str, Any]:
    model = model_module.Kronos.from_pretrained(model_name)
    tokenizer = model_module.KronosTokenizer.from_pretrained(tokenizer_name)
    model_module.KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context=max_context,
    )
    return {
        "model_loaded": True,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "max_context": max_context,
    }


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise KronosRuntimeError(
            "PyTorch is not installed. Follow docs/kronos-runtime.md to install the CUDA wheel."
        ) from exc


def _normalize_python_version(
    python_version: tuple[int, ...] | None,
) -> tuple[int, ...]:
    version = python_version or sys.version_info[:3]
    return tuple(int(part) for part in version[:3])


def _parse_cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    try:
        prefix, index = device.split(":", maxsplit=1)
    except ValueError as exc:
        raise KronosRuntimeError("CUDA device must be 'cuda' or 'cuda:N'.") from exc
    if prefix != "cuda":
        raise KronosRuntimeError("CUDA device must be 'cuda' or 'cuda:N'.")
    try:
        return int(index)
    except ValueError as exc:
        raise KronosRuntimeError("CUDA device index must be an integer.") from exc
