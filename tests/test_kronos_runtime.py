from __future__ import annotations

from pathlib import Path

import pytest

from hourly_prediction.kronos_runtime import KronosRuntimeError, check_kronos_runtime


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX 3060"

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return 10 * 1024**3, 12 * 1024**3


class FakeTorch:
    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = FakeCuda(available=cuda_available)


def write_fake_kronos_model(repo_path: Path, *, include_predictor: bool = True) -> None:
    model_path = repo_path / "model"
    model_path.mkdir(parents=True)
    predictor = (
        "\nclass KronosPredictor:\n"
        "    def __init__(self, *, model, tokenizer, device, max_context):\n"
        "        self.model = model\n"
        "        self.tokenizer = tokenizer\n"
        "        self.device = device\n"
        "        self.max_context = max_context\n"
        if include_predictor
        else ""
    )
    (model_path / "__init__.py").write_text(
        "class Kronos:\n"
        "    loaded_name = None\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, name):\n"
        "        cls.loaded_name = name\n"
        "        return cls()\n"
        "\nclass KronosTokenizer:\n"
        "    loaded_name = None\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, name):\n"
        "        cls.loaded_name = name\n"
        "        return cls()\n"
        f"{predictor}",
        encoding="utf-8",
    )


def write_fake_kronos_submodule(repo_path: Path) -> None:
    model_path = repo_path / "model"
    model_path.mkdir(parents=True)
    (model_path / "__init__.py").write_text("", encoding="utf-8")
    (model_path / "kronos.py").write_text(
        "class Kronos:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, name):\n"
        "        return cls()\n"
        "\nclass KronosTokenizer:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, name):\n"
        "        return cls()\n"
        "\nclass KronosPredictor:\n"
        "    def __init__(self, *, model, tokenizer, device, max_context):\n"
        "        pass\n",
        encoding="utf-8",
    )


def test_cuda_runtime_check_imports_fake_kronos_model(tmp_path) -> None:
    repo_path = tmp_path / "Kronos"
    write_fake_kronos_model(repo_path)

    report = check_kronos_runtime(
        kronos_repo_path=repo_path,
        device="cuda:0",
        mode="cuda",
        torch_module=FakeTorch(),
        python_version=(3, 12, 0),
    )

    assert report["python"] == "3.12.0"
    assert report["cuda_available"] is True
    assert report["device"] == "cuda:0"
    assert report["device_name"] == "NVIDIA GeForce RTX 3060"
    assert report["vram_total_gb"] == 12.0
    assert report["kronos_classes"] == [
        "Kronos",
        "KronosTokenizer",
        "KronosPredictor",
    ]


def test_cuda_runtime_check_accepts_kronos_submodule_exports(tmp_path) -> None:
    repo_path = tmp_path / "Kronos"
    write_fake_kronos_submodule(repo_path)

    report = check_kronos_runtime(
        kronos_repo_path=repo_path,
        device="cuda:0",
        mode="cuda",
        torch_module=FakeTorch(),
        python_version=(3, 12, 0),
    )

    assert report["kronos_classes"] == [
        "Kronos",
        "KronosTokenizer",
        "KronosPredictor",
    ]


def test_model_mode_loads_fake_model_and_tokenizer(tmp_path) -> None:
    repo_path = tmp_path / "Kronos"
    write_fake_kronos_model(repo_path)

    report = check_kronos_runtime(
        kronos_repo_path=repo_path,
        device="cuda:0",
        mode="model",
        torch_module=FakeTorch(),
        python_version=(3, 12, 0),
        model_name="local/model",
        tokenizer_name="local/tokenizer",
        max_context=512,
    )

    assert report["model_loaded"] is True
    assert report["model_name"] == "local/model"
    assert report["tokenizer_name"] == "local/tokenizer"
    assert report["max_context"] == 512


def test_missing_kronos_repo_fails(tmp_path) -> None:
    with pytest.raises(KronosRuntimeError, match="Kronos repo path does not exist"):
        check_kronos_runtime(
            kronos_repo_path=tmp_path / "missing",
            device="cuda:0",
            mode="cuda",
            torch_module=FakeTorch(),
            python_version=(3, 12, 0),
        )


def test_missing_required_kronos_class_fails(tmp_path) -> None:
    repo_path = tmp_path / "Kronos"
    write_fake_kronos_model(repo_path, include_predictor=False)

    with pytest.raises(KronosRuntimeError, match="KronosPredictor"):
        check_kronos_runtime(
            kronos_repo_path=repo_path,
            device="cuda:0",
            mode="cuda",
            torch_module=FakeTorch(),
            python_version=(3, 12, 0),
        )


def test_cuda_unavailable_fails(tmp_path) -> None:
    repo_path = tmp_path / "Kronos"
    write_fake_kronos_model(repo_path)

    with pytest.raises(KronosRuntimeError, match="CUDA is required"):
        check_kronos_runtime(
            kronos_repo_path=repo_path,
            device="cuda:0",
            mode="cuda",
            torch_module=FakeTorch(cuda_available=False),
            python_version=(3, 12, 0),
        )


def test_python_312_is_required(tmp_path) -> None:
    repo_path = tmp_path / "Kronos"
    write_fake_kronos_model(repo_path)

    with pytest.raises(KronosRuntimeError, match="Python 3.12 is required"):
        check_kronos_runtime(
            kronos_repo_path=repo_path,
            device="cuda:0",
            mode="cuda",
            torch_module=FakeTorch(),
            python_version=(3, 14, 5),
        )
