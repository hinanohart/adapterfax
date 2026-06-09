"""Performance-regression guard: the audit must use the small Gram route and
never a wide economy SVD of the p×(N·r) factor stack.

A wide SVD was measured at ~5.6s/layer vs ~0.2s/layer for the Gram-eigh route;
this test fails loudly if any code path reaches for ``np.linalg.svd`` during an
audit, so the regression cannot slip back in.
"""

from __future__ import annotations

import numpy as np
import pytest

from adapterfax import core, synthetic


def test_audit_never_calls_wide_svd(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.linalg.svd was called — wide SVD path regression")

    monkeypatch.setattr(np.linalg, "svd", _boom)
    adapters, _ = synthetic.plant_redundant_adapters(
        400, 4, 0.5, shared_groups=((0, 1, 2),), n_adapters=8, seed=1
    )
    # must complete without ever touching np.linalg.svd
    rep = core.audit(adapters)
    assert rep.active_mode == "full"


def test_gram_is_small_square() -> None:
    # the per-layer Gram is (N·r)×(N·r), not p×(N·r): confirm the route stays tall.
    adapters, _ = synthetic.plant_redundant_adapters(
        500, 4, 0.5, shared_groups=((0, 1),), n_adapters=6, seed=2
    )
    rep = core.audit(adapters)
    lr = rep.per_layer[0]
    assert lr.m == 6 * 4
    assert lr.m < lr.p  # tall stack invariant
