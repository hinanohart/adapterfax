"""Tests for the numpy-native RMT primitives."""

from __future__ import annotations

import math

import numpy as np

from adapterfax import rmt


def test_mp_edges_closed_form() -> None:
    lo, hi = rmt.mp_edges(0.5, 0.25)
    assert math.isclose(hi, 0.5 * (1.5) ** 2)
    assert math.isclose(lo, 0.5 * (0.5) ** 2)


def test_sigma2_recovery_pure_noise() -> None:
    rng = np.random.default_rng(0)
    p, m = 800, 200
    e = np.sort(
        np.linalg.eigvalsh((lambda z: (z.T @ z) / p)(rng.standard_normal((p, m)) * np.sqrt(0.5)))
    )[::-1]
    s2 = rmt.estimate_sigma2(np.asarray(e, dtype=np.float64), m / p)
    assert abs(s2 - 0.5) < 0.02


def test_sigma2_median_matches_mean_on_noise() -> None:
    rng = np.random.default_rng(1)
    p, m = 800, 200
    z = rng.standard_normal((p, m)) * np.sqrt(0.7)
    e = np.asarray(np.sort(np.linalg.eigvalsh((z.T @ z) / p))[::-1], dtype=np.float64)
    s_mean = rmt.estimate_sigma2(e, m / p)
    s_med = rmt.estimate_sigma2_median(e, m / p)
    assert abs(s_mean - s_med) < 0.05


def test_ks_pvalue_monotone() -> None:
    assert rmt.ks_pvalue(0.0, 100) == 1.0
    assert rmt.ks_pvalue(0.5, 100) < rmt.ks_pvalue(0.05, 100)


def test_mp_goodness_of_fit_pure_noise_high_p() -> None:
    rng = np.random.default_rng(2)
    p, m = 800, 200
    z = rng.standard_normal((p, m)) * np.sqrt(0.5)
    e = np.asarray(np.sort(np.linalg.eigvalsh((z.T @ z) / p))[::-1], dtype=np.float64)
    s2 = rmt.estimate_sigma2(e, m / p)
    _, ksp = rmt.mp_goodness_of_fit(e, s2, m / p)
    assert ksp >= 0.05


def test_noise_edge_unit_deterministic_and_cached() -> None:
    a = rmt.noise_edge_unit(800, 200)
    b = rmt.noise_edge_unit(800, 200)
    assert a == b
    assert a > (1.0 + math.sqrt(200 / 800)) ** 2 * 0.99  # near/above MP edge for unit var


def test_classify_directions_threshold_positive() -> None:
    rng = np.random.default_rng(3)
    p, m = 600, 120
    z = rng.standard_normal((p, m)) * np.sqrt(0.3)
    e = np.asarray(np.sort(np.linalg.eigvalsh((z.T @ z) / p))[::-1], dtype=np.float64)
    s2 = rmt.estimate_sigma2(e, m / p)
    mask, thr = rmt.classify_directions(e, s2, p, m)
    assert thr > 0
    assert mask.shape == e.shape


def test_tw_soft_edge_anticonservative_note() -> None:
    # the analytic edge should sit just above the asymptotic MP edge
    s2, p, m = 0.5, 800, 200
    _, hi = rmt.mp_edges(s2, m / p)
    assert rmt.tw_soft_edge(s2, p, m) > hi
