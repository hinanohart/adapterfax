"""Cast adapter layers to column-space factors and apply gauge/scale normalization.

``Y = (α/r)·B`` is the column-space factor whose span equals that of
``ΔW = B @ A`` (we never materialize ``ΔW``).  Per-adapter unit-Frobenius
normalization is the default so that a high-α adapter cannot manufacture a
spike purely from its scale gauge.
"""

from __future__ import annotations

import math

import numpy as np

from .model import FloatArray, LoraLayer

NORMALIZERS = ("noise", "frobenius", "none")


def _robust_noise_scale(y: FloatArray) -> float:
    """Robust per-adapter noise scale = sqrt(median squared singular value).

    Most of an adapter's ``r`` directions are noise; the median singular value
    therefore tracks the noise level and is *not* corrupted by a sparse strong
    signal direction.  Normalizing by it equalizes the noise level across
    adapters (so the stacked factor stays homoscedastic and Marchenko–Pastur
    applies) while remaining invariant to the alpha/scale gauge.
    """
    g = y.T @ y  # r x r, cheap
    ev = np.clip(np.linalg.eigvalsh(g), 0.0, None)
    med = float(np.median(ev))
    return math.sqrt(med) if med > 0.0 else float(np.linalg.norm(y))


def to_factor(layer: LoraLayer, normalize: str = "noise") -> FloatArray:
    """Return the column-space factor ``Y ∈ ℝ^{p×r}`` for one layer.

    * ``lora``  : ``Y = (α/r)·B``.
    * ``dora``  : same, with the per-output magnitude folded in (the audit looks
      at the *direction* matrix scaled by magnitude).
    * ``ia3``   : ``B`` is already the ``p×1`` recast of the diagonal scale.

    ``normalize='noise'`` (default) divides by the robust noise scale so the
    stacked factor is homoscedastic; ``'frobenius'`` divides by the total
    Frobenius norm (legacy); ``'none'`` leaves the raw gauge.
    """
    if normalize not in NORMALIZERS:
        raise ValueError(f"unknown normalize {normalize!r}; expected {NORMALIZERS}")

    y = (layer.alpha / layer.rank) * layer.B.astype(np.float64, copy=False)

    if layer.kind == "dora" and layer.dora_magnitude is not None:
        mag = layer.dora_magnitude.astype(np.float64, copy=False)
        if mag.shape[0] != y.shape[0]:
            raise ValueError(
                f"layer {layer.name!r}: dora_magnitude length {mag.shape[0]} "
                f"!= out_dim {y.shape[0]}"
            )
        y = y * mag[:, None]

    if normalize == "noise" and y.shape[1] >= 2:
        scale = _robust_noise_scale(y)
        if scale > 0.0:
            y = y / scale
    elif normalize == "frobenius":
        fro = float(np.linalg.norm(y))
        if fro > 0.0:
            y = y / fro
    return np.asarray(y, dtype=np.float64)


def stack_layer(
    factors: list[FloatArray], names: list[str]
) -> tuple[FloatArray, list[tuple[str, int, int]]]:
    """Horizontally stack per-adapter factors of one layer into ``Y ∈ ℝ^{p×Σr}``.

    Returns the stacked matrix and a column-ownership index
    ``[(adapter_name, col_start, col_end), ...]`` so directions can be traced
    back to the adapters that load on them.
    """
    if not factors:
        raise ValueError("no factors to stack")
    p = factors[0].shape[0]
    for f, nm in zip(factors, names, strict=True):
        if f.shape[0] != p:
            raise ValueError(f"adapter {nm!r}: out_dim {f.shape[0]} != {p} (layer mismatch)")
    ownership: list[tuple[str, int, int]] = []
    col = 0
    for f, nm in zip(factors, names, strict=True):
        ownership.append((nm, col, col + f.shape[1]))
        col += f.shape[1]
    return np.concatenate(factors, axis=1).astype(np.float64), ownership
