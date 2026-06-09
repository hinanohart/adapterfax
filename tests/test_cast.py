"""Cast: noise normalization, gauge invariance, DoRA/(IA)3 handling."""

from __future__ import annotations

import numpy as np

from adapterfax import cast
from adapterfax.model import LoraLayer


def test_noise_normalization_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    b = rng.standard_normal((100, 6))
    l1 = LoraLayer(name="x", B=b, rank=6, alpha=6.0)
    l2 = LoraLayer(name="x", B=b * 5.0, rank=6, alpha=6.0)
    y1 = cast.to_factor(l1, normalize="noise")
    y2 = cast.to_factor(l2, normalize="noise")
    assert np.allclose(y1, y2, atol=1e-9)


def test_alpha_gauge_invariance_under_noise_norm() -> None:
    rng = np.random.default_rng(1)
    b = rng.standard_normal((80, 4))
    y_lo = cast.to_factor(LoraLayer(name="x", B=b, rank=4, alpha=4.0), normalize="noise")
    y_hi = cast.to_factor(LoraLayer(name="x", B=b, rank=4, alpha=32.0), normalize="noise")
    assert np.allclose(y_lo, y_hi, atol=1e-9)


def test_frobenius_unit_norm() -> None:
    rng = np.random.default_rng(2)
    b = rng.standard_normal((50, 5))
    y = cast.to_factor(LoraLayer(name="x", B=b, rank=5, alpha=5.0), normalize="frobenius")
    assert abs(float(np.linalg.norm(y)) - 1.0) < 1e-9


def test_ia3_recast_rank1() -> None:
    vec = np.ones((30, 1)) * 2.0
    y = cast.to_factor(LoraLayer(name="x", B=vec, rank=1, alpha=1.0, kind="ia3"))
    assert y.shape == (30, 1)


def test_stack_layer_ownership() -> None:
    fa = np.ones((10, 2))
    fb = np.ones((10, 3))
    y, own = cast.stack_layer([fa, fb], ["a", "b"])
    assert y.shape == (10, 5)
    assert own == [("a", 0, 2), ("b", 2, 5)]


def test_stack_layer_dim_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="layer mismatch"):
        cast.stack_layer([np.ones((10, 2)), np.ones((8, 2))], ["a", "b"])
