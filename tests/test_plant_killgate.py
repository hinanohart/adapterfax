"""Fast CI mirror of the S0 plant kill-gate: the RMT core must recover
supercritical spikes and keep pure-noise within the bulk."""

from __future__ import annotations

import math

import numpy as np

from adapterfax import rmt, synthetic

_MULTS = (0.5, 0.9, 1.1, 1.5, 2.0)


def _eigs(y: np.ndarray) -> np.ndarray:
    return np.asarray(np.sort(np.linalg.eigvalsh((y.T @ y) / y.shape[0]))[::-1], dtype=np.float64)


def test_supercritical_spikes_recovered() -> None:
    p, na, r, s2 = 600, 40, 4, 0.5
    n_super = sum(1 for mu in _MULTS if mu >= 1.5)
    hits = 0
    trials = 60
    rng = np.random.default_rng(0)
    for _ in range(trials):
        y, gt = synthetic.plant_spikes(p, na, r, s2, _MULTS, seed=int(rng.integers(1 << 30)))
        e = _eigs(y)
        s2h = rmt.estimate_sigma2(e, gt.gamma)
        mask, _ = rmt.classify_directions(e, s2h, gt.p, gt.m)
        hits += int(np.sum(mask[:n_super]))
    assert hits / (trials * n_super) >= 0.95


def test_pure_noise_low_fpr() -> None:
    p, m, s2 = 600, 150, 0.5
    gamma = m / p
    rng = np.random.default_rng(1)
    fp = 0
    trials = 300
    for _ in range(trials):
        z = rng.standard_normal((p, m)) * math.sqrt(s2)
        e = _eigs(z)
        s2h = rmt.estimate_sigma2(e, gamma)
        mask, _ = rmt.classify_directions(e, s2h, p, m)
        fp += int(bool(np.any(mask)))
    assert fp / trials <= 0.08  # loose bound for the small-trial CI mirror


def test_sigma2_recovered_within_tolerance() -> None:
    y, gt = synthetic.plant_spikes(800, 50, 4, 0.5, _MULTS, seed=7)
    e = _eigs(y)
    assert abs(rmt.estimate_sigma2(e, gt.gamma) - 0.5) < 0.02
