"""Load LoRA / DoRA / (IA)³ adapters from ``.safetensors`` into the data model.

Strict and fail-loud: recognized factor keys are paired into layers; any
leftover ``*.weight`` key that cannot be paired raises, rather than being
silently dropped.  Supports PEFT (``lora_A`` / ``lora_B``) and kohya/diffusers
(``lora_down`` / ``lora_up`` with optional ``.alpha``) naming, DoRA magnitude
vectors, and (IA)³ diagonal scales.
"""

from __future__ import annotations

import os

import numpy as np
from safetensors.numpy import load_file

from .model import Adapter, FloatArray, LoraLayer

_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight")
_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight")
_ALPHA_SUFFIXES = (".alpha",)
_DORA_SUFFIXES = (".lora_magnitude_vector", ".lora_magnitude_vector.weight", ".dora_magnitude")
_IA3_SUFFIXES = (".ia3_l", ".ia3_l.weight")


def _match(key: str, suffixes: tuple[str, ...]) -> str | None:
    for s in suffixes:
        if key.endswith(s):
            return key[: -len(s)]
    return None


def _scalar(arr: FloatArray) -> float:
    return float(np.asarray(arr).reshape(-1)[0])


def load_adapter_file(path: str, *, name: str | None = None) -> Adapter:
    """Load a single ``.safetensors`` adapter file into an :class:`Adapter`."""
    tensors = load_file(path)
    bases: dict[str, dict[str, FloatArray]] = {}
    alphas: dict[str, float] = {}
    consumed: set[str] = set()

    for key, arr in tensors.items():
        a = np.asarray(arr, dtype=np.float64)
        if (base := _match(key, _B_SUFFIXES)) is not None:
            bases.setdefault(base, {})["B"] = a
            consumed.add(key)
        elif (base := _match(key, _A_SUFFIXES)) is not None:
            bases.setdefault(base, {})["A"] = a
            consumed.add(key)
        elif (base := _match(key, _DORA_SUFFIXES)) is not None:
            bases.setdefault(base, {})["mag"] = a.reshape(-1)
            consumed.add(key)
        elif (base := _match(key, _IA3_SUFFIXES)) is not None:
            bases.setdefault(base, {})["ia3"] = a.reshape(-1)
            consumed.add(key)
        elif (base := _match(key, _ALPHA_SUFFIXES)) is not None:
            alphas[base] = _scalar(a)
            consumed.add(key)

    leftover = [k for k in tensors if k.endswith(".weight") and k not in consumed]
    if leftover:
        raise ValueError(
            f"{path}: unrecognized/unpaired weight keys (strict allowlist): "
            f"{sorted(leftover)[:5]}{' ...' if len(leftover) > 5 else ''}"
        )

    layers: dict[str, LoraLayer] = {}
    for base, parts in bases.items():
        if "ia3" in parts:
            vec = parts["ia3"]
            b = (vec - 1.0).reshape(-1, 1)  # delta from identity scale
            layers[base] = LoraLayer(name=base, B=b, rank=1, alpha=1.0, kind="ia3")
            continue
        if "B" not in parts:
            raise ValueError(f"{path}: layer {base!r} has no up/B factor")
        b = parts["B"]
        if b.ndim != 2:
            raise ValueError(f"{path}: layer {base!r} B must be 2-D, got {b.shape}")
        rank = int(b.shape[1])
        alpha = alphas.get(base, float(rank))
        kind = "dora" if "mag" in parts else "lora"
        layers[base] = LoraLayer(
            name=base,
            B=b,
            rank=rank,
            alpha=alpha,
            kind=kind,
            A=parts.get("A"),
            dora_magnitude=parts.get("mag") if kind == "dora" else None,
        )

    if not layers:
        raise ValueError(f"{path}: no LoRA/DoRA/(IA)3 layers found")
    return Adapter(name=name or os.path.basename(path), layers=layers)


def load_adapters(paths: list[str]) -> list[Adapter]:
    """Load several adapter files (one :class:`Adapter` each)."""
    if not paths:
        raise ValueError("no adapter paths supplied")
    return [load_adapter_file(p) for p in paths]
