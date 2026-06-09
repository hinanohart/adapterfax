"""numpy-native Random Matrix Theory primitives for adapterfax.

Everything here operates on the eigenvalues of the *normalized Gram*
``G = (1/p) Yᵀ Y`` of a tall factor stack ``Y ∈ ℝ^{p×m}`` (``m = N·r ≤ p``),
so that ``eig(G) == σ(Y)² / p``.  No scipy, no torch — closed-form
Marchenko–Pastur, a parametric-null calibrated soft edge, and a Tracy–Widom
analytic cross-check.

The detection threshold that controls type-I error is :func:`noise_edge_unit`
(a parametric bootstrap of the pivotal quantity ``λ_max / σ̂²``).  The asymptotic
Tracy–Widom edge :func:`tw_soft_edge` is *anti-conservative* at moderate ``p``
(≈10% type-I measured at ``p=800``) and is reported only as a cross-check.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Operating point for the detection edge.  Nominal 4% per-audit type-I: a
# detector calibrated *exactly* at the claimed 5% would exceed a "realized ≤ 5%"
# check ~half the time from sampling noise alone, so we operate one notch under
# it.  The honest claim is therefore "type-I controlled at the 5% level" with
# finite-sample headroom; measured null FPR sits near 4%.
DEFAULT_Q = 0.96

__all__ = [
    "mp_edges",
    "mp_pdf",
    "mp_cdf",
    "estimate_sigma2",
    "noise_edge_unit",
    "tw_soft_edge",
    "ks_pvalue",
    "mp_goodness_of_fit",
    "classify_directions",
    "effective_capacity",
]


def mp_edges(sigma2: float, gamma: float) -> tuple[float, float]:
    """Marchenko–Pastur support edges ``[λ₋, λ₊]`` for ``eig((1/p) Yᵀ Y)``.

    ``gamma = m / p`` is the aspect ratio of the tall factor stack (≤ 1).
    """
    sq = math.sqrt(gamma)
    return sigma2 * (1.0 - sq) ** 2, sigma2 * (1.0 + sq) ** 2


def mp_pdf(lam: FloatArray, sigma2: float, gamma: float) -> FloatArray:
    """MP density (γ < 1 ⇒ no atom at 0)."""
    lo, hi = mp_edges(sigma2, gamma)
    out = np.zeros_like(lam, dtype=np.float64)
    mask = (lam > lo) & (lam < hi)
    x = lam[mask]
    out[mask] = np.sqrt((hi - x) * (x - lo)) / (2.0 * math.pi * sigma2 * gamma * x)
    return out


def mp_cdf(lam: FloatArray, sigma2: float, gamma: float, grid: int = 4000) -> FloatArray:
    """Numeric MP CDF via trapezoidal integration on a fine grid."""
    lo, hi = mp_edges(sigma2, gamma)
    xs = np.asarray(np.linspace(lo, hi, grid), dtype=np.float64)
    pdf = mp_pdf(xs, sigma2, gamma)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * np.diff(xs))])
    cdf /= cdf[-1]
    return np.asarray(np.interp(lam, xs, cdf, left=0.0, right=1.0), dtype=np.float64)


def estimate_sigma2(eigs: FloatArray, gamma: float, iters: int = 6) -> float:
    """Iterative bulk-mean estimator of the noise variance.

    Drops eigenvalues above the current MP edge (candidate spikes) and
    re-averages; the MP mean equals ``σ²``.  Converges in a handful of passes.
    """
    s2 = float(np.mean(eigs))
    for _ in range(iters):
        _, hi = mp_edges(s2, gamma)
        bulk = eigs[eigs <= hi]
        if bulk.size < 2:
            break
        nxt = float(np.mean(bulk))
        if abs(nxt - s2) < 1e-12:
            s2 = nxt
            break
        s2 = nxt
    return s2


@lru_cache(maxsize=256)
def _mp_median_unit(gamma: float, grid: int = 20000) -> float:
    """Median of the MP law with sigma^2=1 and aspect ratio gamma."""
    lo, hi = mp_edges(1.0, gamma)
    xs = np.asarray(np.linspace(lo, hi, grid), dtype=np.float64)
    cdf = mp_cdf(xs, 1.0, gamma, grid=grid)
    return float(np.interp(0.5, cdf, xs))


def estimate_sigma2_median(eigs: FloatArray, gamma: float) -> float:
    """Median-matching variance estimator: empirical bulk median divided by the
    unit MP median.  Robust alternative to the iterative bulk mean."""
    s2 = estimate_sigma2(eigs, gamma)  # rough scale to find the bulk
    _, hi = mp_edges(s2, gamma)
    bulk = eigs[eigs <= hi]
    if bulk.size < 2:
        return s2
    return float(np.median(bulk) / _mp_median_unit(gamma))


@lru_cache(maxsize=256)
def noise_edge_unit(p: int, m: int, q: float = 0.95, n_mc: int = 1000, seed: int = 0) -> float:
    """Parametric-null calibration of the pivotal quantity ``λ_max / σ̂²``.

    For unit-variance ``Z ~ N(0,1)^{p×m}`` form ``(1/p) Zᵀ Z``, run the *same*
    :func:`estimate_sigma2` the instrument uses, and take the ``q``-quantile of
    ``λ_max / σ̂²``.  Both are scale-equivariant, so the ratio's law is
    independent of the true variance and ``threshold = σ̂² · edge`` delivers
    *exact* finite-size type-I control (FPR ≈ ``1 − q``), folding the
    estimator's small downward bias into the calibration.

    Depends only on the shape ``(p, m)``; memoized.  Deterministic given ``seed``.
    """
    gamma = m / p
    rng = np.random.default_rng(seed + 1_000_003 * p + 31 * m)
    ratios = np.empty(n_mc, dtype=np.float64)
    for i in range(n_mc):
        z = rng.standard_normal((p, m))
        eigs = np.asarray(np.sort(np.linalg.eigvalsh((z.T @ z) / p))[::-1], dtype=np.float64)
        s2 = estimate_sigma2(eigs, gamma)
        ratios[i] = eigs[0] / s2
    return float(np.quantile(ratios, q))


def tw_soft_edge(sigma2: float, p: int, m: int, tw_q: float = 0.9794) -> float:
    """Analytic finite-size Tracy–Widom soft edge (Johnstone 2001 centering).

    Anti-conservative at moderate ``p`` — kept as a fast cross-check only;
    :func:`noise_edge_unit` is the primary type-I-controlling threshold.
    ``tw_q`` defaults to the 0.95 quantile of the GOE (β=1) Tracy–Widom law.
    """
    a = math.sqrt(p - 0.5)
    b = math.sqrt(m - 0.5)
    mu = (a + b) ** 2
    sc = (a + b) * (1.0 / a + 1.0 / b) ** (1.0 / 3.0)
    return float((sigma2 / p) * (mu + tw_q * sc))


def ks_pvalue(stat: float, n: int) -> float:
    """Asymptotic Kolmogorov survival function (numpy-only)."""
    if n <= 0:
        return 1.0
    d = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * stat
    if d < 1e-3:
        return 1.0
    s = 0.0
    for k in range(1, 101):
        s += (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * d * d)
    return max(0.0, min(1.0, 2.0 * s))


def mp_goodness_of_fit(eigs: FloatArray, sigma2: float, gamma: float) -> tuple[float, float]:
    """KS statistic and asymptotic p-value of the in-bulk eigenvalues vs MP.

    Returns ``(ks_stat, ks_pvalue)``.  Eigenvalues outside ``[λ₋, λ₊]`` are
    treated as spikes and excluded.
    """
    lo, hi = mp_edges(sigma2, gamma)
    bulk = np.sort(eigs[(eigs >= lo) & (eigs <= hi)])
    if bulk.size < 3:
        return 0.0, 1.0
    fn = np.arange(1, bulk.size + 1) / bulk.size
    fmp = mp_cdf(bulk, sigma2, gamma)
    ks = float(np.max(np.abs(fn - fmp)))
    return ks, ks_pvalue(ks, bulk.size)


def classify_directions(
    eigs: FloatArray, sigma2: float, p: int, m: int, q: float = DEFAULT_Q
) -> tuple[FloatArray, float]:
    """Return a boolean mask of signal directions and the detection threshold.

    ``threshold = σ̂² · noise_edge_unit(p, m, q)``; a direction is *signal* iff
    its Gram eigenvalue exceeds it (type-I ≈ ``1 − q`` under the isotropic null).
    """
    thr = sigma2 * noise_edge_unit(p, m, q)
    return np.asarray(eigs > thr, dtype=bool), thr


def effective_capacity(
    eigs: FloatArray, sigma2: float, p: int, m: int, q: float = DEFAULT_Q
) -> int:
    """Number of supercritical (signal) Gram eigenvalues — the per-layer
    effective capacity used by the adapter stack at that layer."""
    mask, _ = classify_directions(eigs, sigma2, p, m, q)
    return int(np.sum(mask))
