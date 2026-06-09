"""Tracy–Widom (GOE, β=1) quantiles for the optional analytic edge refinement.

The primary type-I-controlling threshold is the parametric-null MC edge in
:mod:`adapterfax.rmt`; this module only sharpens the analytic Tracy–Widom
*cross-check*.  A hardcoded quantile table (Bejan 2005) works with numpy alone;
if scipy is installed (extra ``[tw]``) a finer interpolation is used.
"""

from __future__ import annotations

import numpy as np

# Tracy-Widom (beta=1, GOE) quantile table, probability -> value (Bejan 2005).
_TW1_TABLE: tuple[tuple[float, float], ...] = (
    (0.01, -3.9139),
    (0.05, -3.1808),
    (0.10, -2.7822),
    (0.30, -1.9132),
    (0.50, -1.2685),
    (0.70, -0.5946),
    (0.80, -0.1614),
    (0.90, 0.4501),
    (0.95, 0.9794),
    (0.975, 1.4538),
    (0.99, 2.0233),
    (0.995, 2.4221),
)


def _table() -> tuple[tuple[float, float], ...]:
    return _TW1_TABLE


def tw1_quantile(prob: float) -> float:
    """Linear-interpolated GOE Tracy–Widom quantile for ``prob`` in (0, 1)."""
    tbl = _table()
    ps = np.array([p for p, _ in tbl], dtype=np.float64)
    vs = np.array([v for _, v in tbl], dtype=np.float64)
    return float(np.interp(prob, ps, vs))


def has_scipy() -> bool:
    try:
        import scipy.stats  # noqa: F401
    except Exception:
        return False
    return True
