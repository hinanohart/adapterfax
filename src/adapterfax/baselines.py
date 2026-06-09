"""Baseline redundancy/capacity measures run alongside adapterfax for honesty.

None of these emit a *cross-adapter inclusion* redundant subset — that gap is
exactly what the dependency census fills (gate G6).  In particular **PARA**
(arXiv:2604.27796) prunes *per-adapter* rank by a global Frobenius-energy
percentile and reports a retained rank per adapter; it never names a subset of
adapters that jointly carry one redundant direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray

__all__ = [
    "effective_rank",
    "para_retained_ranks",
    "cosine_redundancy",
    "ties_sign_agreement",
    "dare_drop_invariance",
    "gavish_donoho_count",
    "BaselineResult",
]


def _singvals(y: FloatArray) -> FloatArray:
    # singular values via the small Gram (m x m) — never a wide SVD.
    g = y.T @ y
    ev = np.clip(np.linalg.eigvalsh(g), 0.0, None)
    sv = np.sqrt(np.sort(ev)[::-1])
    return np.asarray(sv, dtype=np.float64)


def effective_rank(y: FloatArray) -> float:
    """Roy–Vetterli effective rank: ``exp(H(p))`` over normalized singular values."""
    s = _singvals(y)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    h = -float(np.sum(p * np.log(p)))
    return float(np.exp(h))


def para_retained_ranks(
    factors: dict[str, FloatArray], energy_keep: float = 0.99
) -> dict[str, int]:
    """PARA-style data-free prune: a single global singular-value threshold set
    at the ``energy_keep`` Frobenius-energy percentile across *all* adapters,
    then count each adapter's singular values above it.

    Returns a per-adapter retained rank.  Deliberately has no notion of a shared
    cross-adapter direction.
    """
    all_sv: list[float] = []
    per: dict[str, FloatArray] = {}
    for name, y in factors.items():
        sv = _singvals(y)
        per[name] = sv
        all_sv.extend(sv.tolist())
    if not all_sv:
        return {name: 0 for name in factors}
    arr = np.sort(np.array(all_sv, dtype=np.float64))[::-1]
    energy = np.cumsum(arr**2) / np.sum(arr**2)
    cut_idx = int(np.searchsorted(energy, energy_keep))
    cut_idx = min(cut_idx, arr.size - 1)
    thr = float(arr[cut_idx])
    return {name: int(np.sum(sv >= thr)) for name, sv in per.items()}


def cosine_redundancy(factors: dict[str, FloatArray]) -> float:
    """Max pairwise cosine similarity between adapters' top singular directions."""
    tops: list[FloatArray] = []
    for y in factors.values():
        # leading left singular vector via the largest Gram eigenpair
        g = y.T @ y
        _, v = np.linalg.eigh(g)
        top = y @ v[:, -1]
        n = float(np.linalg.norm(top))
        if n > 0:
            tops.append(np.asarray(top / n, dtype=np.float64))
    best = 0.0
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            best = max(best, abs(float(tops[i] @ tops[j])))
    return best


def ties_sign_agreement(factors: dict[str, FloatArray]) -> float:
    """TIES-style mean pairwise sign agreement of the leading directions."""
    tops: list[FloatArray] = []
    for y in factors.values():
        g = y.T @ y
        _, v = np.linalg.eigh(g)
        top = y @ v[:, -1]
        n = float(np.linalg.norm(top))
        if n > 0:
            tops.append(np.asarray(np.sign(top), dtype=np.float64))
    if len(tops) < 2:
        return 0.0
    vals: list[float] = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            vals.append(float(np.mean(tops[i] == tops[j])))
    return float(np.mean(vals))


def dare_drop_invariance(factors: dict[str, FloatArray], drop: float = 0.5) -> float:
    """DARE-style: Frobenius energy retained after random magnitude dropping +
    rescaling, averaged — a coarse redundancy proxy (high ⇒ drop-robust)."""
    rng = np.random.default_rng(0)
    ratios: list[float] = []
    for y in factors.values():
        mask = rng.random(y.shape) >= drop
        kept = (y * mask) / max(1e-12, (1.0 - drop))
        num = float(np.linalg.norm(kept))
        den = float(np.linalg.norm(y))
        if den > 0:
            ratios.append(num / den)
    return float(np.mean(ratios)) if ratios else 0.0


def gavish_donoho_count(y: FloatArray, sigma2: float) -> int:
    """Count singular values above the Gavish–Donoho optimal hard threshold for
    a known noise level (naive single-matrix baseline)."""
    p, m = y.shape
    beta = min(p, m) / max(p, m)
    lam = (2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1))) ** 0.5
    thr = lam * np.sqrt(max(p, m)) * np.sqrt(sigma2)
    return int(np.sum(_singvals(y) > thr))


@dataclass(frozen=True)
class BaselineResult:
    erank: float
    para_retained: dict[str, int]
    para_total: int
    cosine_redundancy: float
    ties_sign_agreement: float
    dare_drop_invariance: float
    gavish_donoho_count: int
