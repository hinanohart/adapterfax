"""Pre-registered sensitivity gates G1..G9 (no threshold tuning).

Each gate exercises one claim against synthetic ground-truth.  ``run_gates``
returns structured results and an overall pass/active-mode used by both the
``adapterfax gate`` CLI and the test suite.  ``fast=True`` shrinks trial counts
for CI; the S4 validation run uses ``fast=False`` (1000 null trials).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import baselines, cast, core, rmt, synthetic
from .model import Adapter, LoraLayer

_MULTS = (0.5, 0.9, 1.1, 1.5, 2.0)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: dict[str, object]


@dataclass(frozen=True)
class GateConfig:
    p: int = 360
    n_adapters: int = 22
    rank: int = 4
    sigma2: float = 0.5
    tpr_trials: int = 50
    null_trials: int = 600
    census_p: int = 480
    census_n: int = 8
    seed: int = 20260609

    @classmethod
    def for_run(cls, fast: bool) -> GateConfig:
        if fast:
            return cls()
        return cls(p=800, n_adapters=50, tpr_trials=400, null_trials=1000)


def _bootstrap_lo(per_trial: np.ndarray, seed: int, n: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    boot = rng.choice(per_trial, size=(n, per_trial.size), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5))


def _eigs(y: np.ndarray) -> np.ndarray:
    return np.asarray(np.sort(np.linalg.eigvalsh((y.T @ y) / y.shape[0]))[::-1], dtype=np.float64)


def g1_tpr(c: GateConfig) -> GateResult:
    rng = np.random.default_rng(c.seed)
    n_super = sum(1 for mu in _MULTS if mu >= 1.5)
    hits = np.empty(c.tpr_trials)
    for t in range(c.tpr_trials):
        y, gt = synthetic.plant_spikes(
            c.p, c.n_adapters, c.rank, c.sigma2, _MULTS, seed=int(rng.integers(1 << 30))
        )
        e = _eigs(y)
        s2 = rmt.estimate_sigma2(e, gt.gamma)
        mask, _ = rmt.classify_directions(e, s2, gt.p, gt.m)
        hits[t] = int(np.sum(mask[:n_super])) / n_super
    tpr = float(hits.mean())
    lo = _bootstrap_lo(hits, c.seed)
    return GateResult(
        "G1_tpr", tpr >= 0.95 and lo >= 0.90, {"tpr": tpr, "ci_lo": lo, "n_supercritical": n_super}
    )


def g2_fpr(c: GateConfig, gamma_override: tuple[int, int] | None = None) -> GateResult:
    rng = np.random.default_rng(c.seed + 7)
    if gamma_override is None:
        p, m = c.p, c.n_adapters * c.rank
    else:
        p, m = gamma_override
    gamma = m / p
    fp = 0
    for _ in range(c.null_trials):
        zmat = rng.standard_normal((p, m)) * math.sqrt(c.sigma2)
        e = _eigs(zmat)
        s2 = rmt.estimate_sigma2(e, gamma)
        mask, _ = rmt.classify_directions(e, s2, p, m)
        if bool(np.any(mask)):
            fp += 1
    fpr = fp / c.null_trials
    zc = 1.96
    n = c.null_trials
    hi = (fpr + zc * zc / (2 * n) + zc * math.sqrt(fpr * (1 - fpr) / n + zc * zc / (4 * n * n))) / (
        1 + zc * zc / n
    )
    return GateResult(
        "G2_fpr", fpr <= 0.05 and hi <= 0.07, {"fpr": fpr, "ci_hi": hi, "p": p, "m": m}
    )


def g3_calibration(c: GateConfig) -> GateResult:
    y, gt = synthetic.plant_spikes(c.p, c.n_adapters, c.rank, c.sigma2, _MULTS, seed=c.seed)
    e = _eigs(y)
    s2 = rmt.estimate_sigma2(e, gt.gamma)
    cap = rmt.effective_capacity(e, s2, gt.p, gt.m)
    err = abs(cap - gt.n_supercritical)
    return GateResult(
        "G3_calibration",
        err <= 1,
        {"estimated": cap, "planted_supercritical": gt.n_supercritical, "err": err},
    )


def g4_estimator_consistency(c: GateConfig) -> GateResult:
    y, gt = synthetic.plant_spikes(c.p, c.n_adapters, c.rank, c.sigma2, _MULTS, seed=c.seed + 3)
    e = _eigs(y)
    cap_mean = rmt.effective_capacity(e, rmt.estimate_sigma2(e, gt.gamma), gt.p, gt.m)
    cap_med = rmt.effective_capacity(e, rmt.estimate_sigma2_median(e, gt.gamma), gt.p, gt.m)
    return GateResult(
        "G4_estimator_consistency",
        abs(cap_mean - cap_med) <= 1,
        {"bulk_mean": cap_mean, "median_match": cap_med},
    )


def g5_determinism(c: GateConfig, reps: int = 20) -> GateResult:
    refs: list[bytes] = []
    for _ in range(reps):
        y, _ = synthetic.plant_spikes(c.p, 20, c.rank, c.sigma2, _MULTS, seed=c.seed)
        e = _eigs(y)
        refs.append(e.tobytes())
    ok = all(r == refs[0] for r in refs)
    return GateResult("G5_determinism", ok, {"reps": reps, "identical": ok})


def _census_subsets(adapters: list[Adapter], tol: float | None = None) -> set[frozenset[str]]:
    rep = core.audit(adapters, active_mode="full", tol=tol)
    return {frozenset(ci.members) for ci in rep.dependency_census}


def g6_census_nontrivial(c: GateConfig) -> GateResult:
    # 3 adapters share one direction; PARA cannot name the subset.
    groups = ((0, 1, 2),)
    adapters, gt = synthetic.plant_redundant_adapters(
        c.census_p, c.rank, c.sigma2, shared_groups=groups, n_adapters=c.census_n, seed=c.seed + 5
    )
    found = _census_subsets(adapters)
    planted = {frozenset(s) for s in gt.redundant_subsets}
    detected = planted.issubset(found)
    # tolerance robustness: stable across a sweep
    factors = {ad.name: cast.to_factor(next(iter(ad.layers.values()))) for ad in adapters}
    para = baselines.para_retained_ranks(factors)
    # live check: PARA structurally returns a per-adapter integer rank, so it can
    # never name a *subset* of adapters; this must hold for the wedge to exist.
    para_names_subset = not all(isinstance(v, int) for v in para.values())
    stable = 0
    sweep = (0.03, 0.0625, 0.10)
    for tol in sweep:
        if planted.issubset(_census_subsets(adapters, tol=tol)):
            stable += 1
    robust = stable >= 2
    return GateResult(
        "G6_census_nontrivial",
        detected and robust and not para_names_subset,
        {
            "planted": [sorted(s) for s in planted],
            "found": [sorted(s) for s in found],
            "tolerance_stable": f"{stable}/{len(sweep)}",
            "para_retained": para,
            "para_emits_subset": para_names_subset,
        },
    )


def g7_scale_invariance(c: GateConfig) -> GateResult:
    groups = ((0, 1, 2),)
    adapters, _ = synthetic.plant_redundant_adapters(
        c.census_p, c.rank, c.sigma2, shared_groups=groups, n_adapters=c.census_n, seed=c.seed + 9
    )
    base = _census_subsets(adapters)
    scaled: list[Adapter] = []

    for ad in adapters:
        ln, layer = next(iter(ad.layers.items()))
        scaled.append(
            Adapter(
                name=ad.name,
                layers={
                    ln: LoraLayer(
                        name=layer.name,
                        B=layer.B * 7.3,
                        rank=layer.rank,
                        alpha=layer.alpha,
                        kind=layer.kind,
                    )
                },
            )
        )
    scaled_sets = _census_subsets(scaled)
    return GateResult(
        "G7_scale_invariance",
        base == scaled_sets,
        {"base": [sorted(s) for s in base], "scaled": [sorted(s) for s in scaled_sets]},
    )


def g8_fpr_second_shape(c: GateConfig) -> GateResult:
    # confirmatory type-I check at a second aspect ratio (gamma=0.5); a smaller
    # shape keeps the eigendecomposition cheap.
    shape = (300, 150) if c.null_trials <= 600 else (400, 200)
    res = g2_fpr(c, gamma_override=shape)
    return GateResult("G8_fpr_second_shape", res.passed, res.detail)


def g9_end_to_end_determinism(c: GateConfig) -> GateResult:
    groups = ((0, 1, 2), (3, 4))
    adapters, _ = synthetic.plant_redundant_adapters(
        c.census_p, c.rank, c.sigma2, shared_groups=groups, n_adapters=c.census_n, seed=c.seed + 11
    )
    j1 = core.audit(adapters, active_mode="full").to_json()
    j2 = core.audit(adapters, active_mode="full").to_json()
    return GateResult("G9_end_to_end_determinism", j1 == j2, {"identical": j1 == j2})


def run_gates(fast: bool = True) -> dict[str, object]:
    c = GateConfig.for_run(fast)
    results = [
        g1_tpr(c),
        g2_fpr(c),
        g3_calibration(c),
        g4_estimator_consistency(c),
        g5_determinism(c),
        g6_census_nontrivial(c),
        g7_scale_invariance(c),
        g8_fpr_second_shape(c),
        g9_end_to_end_determinism(c),
    ]
    by = {r.name: asdict(r) for r in results}
    hard = [r for r in results if r.name != "G6_census_nontrivial"]
    g6 = next(r for r in results if r.name == "G6_census_nontrivial")
    hard_pass = all(r.passed for r in hard)
    if hard_pass and g6.passed:
        active_mode, all_pass = "full", True
    elif hard_pass and not g6.passed:
        active_mode, all_pass = "lite", True  # G6 RED -> P-LITE (honest-DONE, not kill)
    else:
        active_mode, all_pass = "full", False
    return {"gates": by, "active_mode": active_mode, "all_pass": all_pass, "config_fast": fast}
