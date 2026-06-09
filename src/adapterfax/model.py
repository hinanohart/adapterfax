"""In-memory data model shared by the loader, the synthetic generator and audit.

An :class:`Adapter` is a named collection of :class:`LoraLayer` objects.  The
audit never materializes ``ΔW = B @ A`` (≈ p·q entries); it works on the
column-space factor ``Y = (α/r)·B`` produced by :mod:`adapterfax.cast`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

VALID_KINDS = ("lora", "dora", "ia3")


@dataclass(frozen=True)
class LoraLayer:
    """One adapted layer of one adapter.

    ``B`` is the up/output factor (``p×r``).  ``A`` (the down factor, ``r×q``) is
    optional and unused by the column-space audit.  For ``kind='dora'`` the
    direction magnitude vector ``dora_magnitude`` (length ``p``) folds into the
    factor; for ``kind='ia3'`` ``B`` is the diagonal scale recast to ``p×1``.
    """

    name: str
    B: FloatArray
    rank: int
    alpha: float
    kind: str = "lora"
    A: FloatArray | None = None
    dora_magnitude: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown adapter kind {self.kind!r}; expected {VALID_KINDS}")
        if self.B.ndim != 2:
            raise ValueError(f"layer {self.name!r}: B must be 2-D, got shape {self.B.shape}")
        if self.rank <= 0:
            raise ValueError(f"layer {self.name!r}: rank must be positive, got {self.rank}")

    @property
    def out_dim(self) -> int:
        return int(self.B.shape[0])


@dataclass(frozen=True)
class Adapter:
    """A named adapter: a mapping ``layer_name -> LoraLayer``."""

    name: str
    layers: Mapping[str, LoraLayer]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError(f"adapter {self.name!r} has no layers")

    def layer_names(self) -> list[str]:
        return list(self.layers.keys())
