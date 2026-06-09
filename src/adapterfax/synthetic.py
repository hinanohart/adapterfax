"""First-class synthetic plant generators with known ground-truth.

Two generators:

* :func:`plant_spikes` — a single stacked factor matrix with planted
  signal directions of controlled BBP strength plus isotropic noise.  Drives
  the RMT gates (G1 TPR, G2 FPR, G3 calibration, G5 determinism, G7 scale, G8).

* :func:`plant_redundant_adapters` — a set of :class:`Adapter` objects in which
  named *shared* directions appear across chosen adapter subsets, creating
  genuine cross-adapter inclusion redundancy that per-adapter rank pruning
  (PARA) cannot see.  Drives the census non-triviality gate (G6 / OQ4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import Adapter, FloatArray, LoraLayer


@dataclass(frozen=True)
class SpikeGroundTruth:
    """Planted spike metadata for one stacked factor matrix."""

    p: int
    m: int
    gamma: float
    sigma2: float
    theta_star: float
    mults: tuple[float, ...]
    energies: tuple[float, ...]

    @property
    def n_supercritical(self) -> int:
        return sum(1 for mu in self.mults if mu >= 1.5)

    @property
    def n_signal(self) -> int:
        # directions expected to clear the bulk edge (BBP supercritical, μ>1)
        return sum(1 for mu in self.mults if mu > 1.0)


def plant_spikes(
    p: int,
    n_adapters: int,
    rank: int,
    sigma2: float,
    mults: tuple[float, ...],
    *,
    seed: int,
) -> tuple[FloatArray, SpikeGroundTruth]:
    """Build ``Y ∈ ℝ^{p×m}`` (``m = n_adapters·rank``) = isotropic ``N(0,σ²)``
    noise + planted signal directions whose normalized-Gram eigenvalue is
    ``(mult·θ*)²`` with ``θ*² = σ²·√γ`` (the BBP eigenvalue threshold)."""
    rng = np.random.default_rng(seed)
    m = n_adapters * rank
    gamma = m / p
    theta_star2 = sigma2 * math.sqrt(gamma)
    y = rng.standard_normal((p, m)) * math.sqrt(sigma2)
    u, _ = np.linalg.qr(rng.standard_normal((p, len(mults))))
    v, _ = np.linalg.qr(rng.standard_normal((m, len(mults))))
    energies: list[float] = []
    for j, mu in enumerate(mults):
        ell = (mu**2) * theta_star2
        energies.append(ell)
        s = math.sqrt(p * ell)
        y += s * np.outer(u[:, j], v[:, j])
    gt = SpikeGroundTruth(
        p=p,
        m=m,
        gamma=gamma,
        sigma2=sigma2,
        theta_star=math.sqrt(theta_star2),
        mults=tuple(mults),
        energies=tuple(energies),
    )
    return np.asarray(y, dtype=np.float64), gt


@dataclass(frozen=True)
class RedundancyGroundTruth:
    """Which adapter subsets share a planted direction (truly redundant)."""

    redundant_subsets: tuple[tuple[str, ...], ...]
    layer_name: str


def plant_redundant_adapters(
    p: int,
    rank: int,
    sigma2: float,
    *,
    shared_groups: tuple[tuple[int, ...], ...],
    n_adapters: int,
    signal_strength: float = 25.0,
    seed: int,
    layer_name: str = "layer0",
) -> tuple[list[Adapter], RedundancyGroundTruth]:
    """Build ``n_adapters`` rank-``rank`` adapters over one layer such that each
    group in ``shared_groups`` (indices into the adapters) shares one common
    column-space direction — a cross-adapter inclusion redundancy invisible to
    per-adapter rank pruning.  ``signal_strength`` scales the shared direction
    well above the noise floor so the redundancy is unambiguous.
    """
    rng = np.random.default_rng(seed)
    # one shared unit direction in R^p per group
    shared_dirs = [rng.standard_normal(p) for _ in shared_groups]
    shared_dirs = [d / np.linalg.norm(d) for d in shared_dirs]

    adapters: list[Adapter] = []
    names = [f"adapter{i}" for i in range(n_adapters)]
    for i in range(n_adapters):
        # private random factor B (p x rank), small isotropic energy
        b = rng.standard_normal((p, rank)) * math.sqrt(sigma2)
        # inject any shared directions this adapter belongs to into column 0
        for g, group in enumerate(shared_groups):
            if i in group:
                b[:, 0] += signal_strength * shared_dirs[g]
        layer = LoraLayer(name=layer_name, B=b.astype(np.float64), rank=rank, alpha=float(rank))
        adapters.append(Adapter(name=names[i], layers={layer_name: layer}))

    redundant = tuple(tuple(names[i] for i in group) for group in shared_groups if len(group) >= 2)
    return adapters, RedundancyGroundTruth(redundant_subsets=redundant, layer_name=layer_name)
