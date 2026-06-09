"""Loader: PEFT + kohya naming, DoRA, (IA)3, and fail-loud on bad keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adapterfax import loader

_RNG = np.random.default_rng(0)


def _randn(*shape: int) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _save(tmp_path: Path, name: str, tensors: dict[str, np.ndarray]) -> str:
    path = str(tmp_path / name)
    save_file(tensors, path)
    return path


def test_peft_naming(tmp_path: Path) -> None:
    p, q, r = 64, 32, 4
    t = {
        "base_model.layer1.lora_A.weight": _randn(r, q),
        "base_model.layer1.lora_B.weight": _randn(p, r),
    }
    ad = loader.load_adapter_file(_save(tmp_path, "a.safetensors", t))
    assert "base_model.layer1" in ad.layers
    lyr = ad.layers["base_model.layer1"]
    assert lyr.kind == "lora" and lyr.rank == r and lyr.out_dim == p


def test_kohya_naming_with_alpha(tmp_path: Path) -> None:
    p, q, r = 48, 24, 8
    t = {
        "lora_unet_block.lora_down.weight": _randn(r, q),
        "lora_unet_block.lora_up.weight": _randn(p, r),
        "lora_unet_block.alpha": np.array(16.0, dtype=np.float32),
    }
    ad = loader.load_adapter_file(_save(tmp_path, "k.safetensors", t))
    lyr = ad.layers["lora_unet_block"]
    assert lyr.alpha == 16.0 and lyr.rank == r


def test_dora_magnitude(tmp_path: Path) -> None:
    p, q, r = 40, 20, 4
    t = {
        "m.lora_A.weight": _randn(r, q),
        "m.lora_B.weight": _randn(p, r),
        "m.lora_magnitude_vector": _RNG.random(p).astype(np.float32),
    }
    ad = loader.load_adapter_file(_save(tmp_path, "d.safetensors", t))
    assert ad.layers["m"].kind == "dora"
    assert ad.layers["m"].dora_magnitude is not None


def test_ia3(tmp_path: Path) -> None:
    t = {"attn.ia3_l": _RNG.random(50).astype(np.float32)}
    ad = loader.load_adapter_file(_save(tmp_path, "i.safetensors", t))
    lyr = ad.layers["attn"]
    assert lyr.kind == "ia3" and lyr.rank == 1 and lyr.out_dim == 50


def test_fail_loud_on_unpaired_weight(tmp_path: Path) -> None:
    t = {"mystery.weight": _randn(10, 10)}
    with pytest.raises(ValueError, match="unrecognized|no LoRA"):
        loader.load_adapter_file(_save(tmp_path, "bad.safetensors", t))


def test_empty_paths_raises() -> None:
    with pytest.raises(ValueError, match="no adapter paths"):
        loader.load_adapters([])
