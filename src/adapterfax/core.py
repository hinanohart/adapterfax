"""The audit entry point: stack adapters per layer, run the RMT pipeline, and
assemble a :class:`~adapterfax.report.Report`.

Mandatory Gram route: per layer we form ``G = (1/p) Yᵀ Y`` (``m×m``) and call
``eigh`` — never a wide economy SVD of the ``p×(N·r)`` factor stack.
"""

from __future__ import annotations

import numpy as np

from . import __version__, baselines, cast, census, rmt
from .model import Adapter, FloatArray
from .report import (
    SCHEMA_VERSION,
    Calibration,
    Direction,
    LayerReport,
    Report,
)

AGGREGATIONS = ("layerwise", "global")
ESTIMATORS = ("bulk_mean", "median_match")


def _group_layers(adapters: list[Adapter]) -> dict[str, list[str]]:
    """Map each layer name to the adapters that adapt it (order preserved)."""
    layers: dict[str, list[str]] = {}
    for ad in adapters:
        for ln in ad.layer_names():
            layers.setdefault(ln, []).append(ad.name)
    return layers


def _eigh_desc(g: FloatArray) -> tuple[FloatArray, FloatArray]:
    w, v = np.linalg.eigh(g)
    order = np.argsort(w)[::-1]
    return np.asarray(w[order], dtype=np.float64), np.asarray(v[:, order], dtype=np.float64)


def audit(
    adapters: list[Adapter],
    *,
    tol: float | None = None,
    aggregation: str = "layerwise",
    estimator: str = "bulk_mean",
    normalize: str = "noise",
    q: float = rmt.DEFAULT_Q,
    tw_refine: bool = False,
    active_mode: str = "full",
    seed: int = 0,
    provenance: dict[str, object] | None = None,
) -> Report:
    """Audit a stack of adapters and return a :class:`Report`.

    Parameters mirror the public CLI.  ``tol=None`` derives the census tolerance
    from the adapter count.  ``active_mode`` controls the claim level (``full``,
    ``lite``, ``mp-bulk-only``, ``erank-gate``, ``synth-only``).
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected {AGGREGATIONS}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected {ESTIMATORS}")
    if not adapters:
        raise ValueError("no adapters supplied")

    by_name = {ad.name: ad for ad in adapters}
    layer_groups = _group_layers(adapters)
    est_fn = rmt.estimate_sigma2 if estimator == "bulk_mean" else rmt.estimate_sigma2_median
    census_on = active_mode in ("full",)

    per_layer: list[LayerReport] = []
    directions: list[Direction] = []
    all_circuits: list[census.Circuit] = []
    factors_all: dict[str, FloatArray] = {}
    cap_per_layer: dict[str, int] = {}
    gammas: list[float] = []
    s2s: list[float] = []
    ks_ps: list[float] = []

    for ln, adapter_names in layer_groups.items():
        factors: list[FloatArray] = []
        for nm in adapter_names:
            layer = by_name[nm].layers[ln]
            factors.append(cast.to_factor(layer, normalize=normalize))
        y, ownership = cast.stack_layer(factors, adapter_names)
        for nm, f in zip(adapter_names, factors, strict=True):
            factors_all[f"{ln}:{nm}"] = f
        p, m = y.shape
        if m > p:
            raise ValueError(
                f"layer {ln!r}: stacked columns {m} exceed rows {p}; the factor "
                f"stack must stay tall (reduce adapters or rank)"
            )
        gamma = m / p
        g = (y.T @ y) / p
        eigs, vecs = _eigh_desc(g)
        s2 = est_fn(eigs, gamma)
        mask, thr = rmt.classify_directions(eigs, s2, p, m, q)
        cap = int(np.sum(mask))
        ks_stat, ks_p = rmt.mp_goodness_of_fit(eigs, s2, gamma)
        _, mp_hi = rmt.mp_edges(s2, gamma)

        # signal eigenvectors for the census
        sig_idx = np.where(mask)[0]
        signal_vecs = vecs[:, sig_idx] if sig_idx.size else np.zeros((m, 0), dtype=np.float64)
        ctol = census.default_tolerance(len(adapter_names)) if tol is None else tol
        circuits = census.dependency_census(signal_vecs, ownership, ctol) if census_on else []
        all_circuits.extend(circuits)

        # rank certificate directions (signal + a few sub-edge for context)
        blocks = {nm: (c0, c1) for nm, c0, c1 in ownership}
        for k in range(m):
            cls = "signal" if mask[k] else "bulk_noise"
            if cls == "bulk_noise" and k >= cap + 3:
                break  # only keep signal + a small bulk margin
            loaders = _loaders(vecs[:, k], blocks, ctol)
            directions.append(
                Direction(
                    layer=ln,
                    eigenvalue=float(eigs[k]),
                    classification=cls,
                    loading_adapters=loaders,
                )
            )

        per_layer.append(
            LayerReport(
                layer=ln,
                p=int(p),
                m=int(m),
                gamma=float(gamma),
                sigma2_hat=float(s2),
                mp_edge=float(mp_hi),
                detect_threshold=float(thr),
                effective_capacity=cap,
                ks_stat=float(ks_stat),
                ks_pvalue=float(ks_p),
                n_circuits=len(circuits),
            )
        )
        cap_per_layer[ln] = cap
        gammas.append(gamma)
        s2s.append(s2)
        ks_ps.append(ks_p)

    cap_used = int(sum(cap_per_layer.values()))
    ci = _bootstrap_ci(list(cap_per_layer.values()), seed=seed)

    base = baselines.BaselineResult(
        erank=_safe(lambda: baselines.effective_rank(_concat(factors_all))),
        para_retained=baselines.para_retained_ranks(factors_all),
        para_total=int(sum(baselines.para_retained_ranks(factors_all).values())),
        cosine_redundancy=baselines.cosine_redundancy(factors_all),
        ties_sign_agreement=baselines.ties_sign_agreement(factors_all),
        dare_drop_invariance=baselines.dare_drop_invariance(factors_all),
        gavish_donoho_count=baselines.gavish_donoho_count(
            _concat(factors_all), float(np.mean(s2s)) if s2s else 1.0
        ),
    )

    calibration = Calibration(
        gamma_mean=float(np.mean(gammas)) if gammas else 0.0,
        sigma2_hat_mean=float(np.mean(s2s)) if s2s else 0.0,
        estimator=estimator,
        edge_method="parametric-null-mc (pivotal lambda_max/sigma2_hat)",
        tw_analytic_crosscheck=bool(tw_refine),
        mp_goodness_of_fit_min_p=float(min(ks_ps)) if ks_ps else 1.0,
        degraded_to=None if active_mode == "full" else active_mode,
        disclaimer=(
            "MP/BBP thresholds are isotropic-iid approximations, no calibration "
            "guarantee for structured rank-r LoRA delta."
        ),
    )

    return Report(
        schema_version=SCHEMA_VERSION,
        active_mode=active_mode,
        headline=_headline(active_mode),
        effective_capacity_used=cap_used,
        effective_capacity_per_layer=cap_per_layer,
        effective_capacity_ci=ci,
        rank_certificate=tuple(directions),
        dependency_census=tuple(all_circuits),
        baselines={
            "erank": base.erank,
            "para_retained_ranks": base.para_retained,
            "para_total": base.para_total,
            "cosine_redundancy": base.cosine_redundancy,
            "ties_sign_agreement": base.ties_sign_agreement,
            "dare_drop_invariance": base.dare_drop_invariance,
            "gavish_donoho_count": base.gavish_donoho_count,
        },
        per_layer=tuple(per_layer),
        calibration=calibration,
        provenance=provenance or {"numpy": np.__version__, "adapterfax": __version__},
    )


def _loaders(vec: FloatArray, blocks: dict[str, tuple[int, int]], tol: float) -> tuple[str, ...]:
    total = float(vec @ vec)
    if total <= 0:
        return ()
    out: list[tuple[str, float]] = []
    for nm, (c0, c1) in blocks.items():
        seg = vec[c0:c1]
        fr = float(seg @ seg) / total
        if fr > tol:
            out.append((nm, fr))
    out.sort(key=lambda t: -t[1])
    return tuple(nm for nm, _ in out)


def _bootstrap_ci(values: list[int], seed: int, n_boot: int = 2000) -> tuple[float, float]:
    if len(values) <= 1:
        v = float(values[0]) if values else 0.0
        return (v, v)
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).sum(axis=1)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _concat(factors_all: dict[str, FloatArray]) -> FloatArray:
    mats = list(factors_all.values())
    p = mats[0].shape[0]
    same = [mm for mm in mats if mm.shape[0] == p]
    return np.concatenate(same, axis=1).astype(np.float64)


def _safe(fn: object) -> float:
    try:
        return float(fn())  # type: ignore[operator]
    except Exception:
        return float("nan")


def _headline(active_mode: str) -> str:
    from .report import HEADLINE

    if active_mode == "full":
        return HEADLINE
    return HEADLINE + f" [degraded mode: {active_mode}]"
