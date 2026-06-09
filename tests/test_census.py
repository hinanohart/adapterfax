"""Census recovers planted cross-adapter redundancy and is quiet on pure noise."""

from __future__ import annotations

from adapterfax import core, synthetic


def _subsets(rep: object) -> set[frozenset[str]]:
    return {frozenset(c.members) for c in rep.dependency_census}  # type: ignore[attr-defined]


def test_census_recovers_two_planted_groups() -> None:
    ad, gt = synthetic.plant_redundant_adapters(
        800, 4, 0.5, shared_groups=((0, 1, 2), (3, 4)), n_adapters=8, seed=7
    )
    rep = core.audit(ad)
    planted = {frozenset(s) for s in gt.redundant_subsets}
    assert planted.issubset(_subsets(rep))


def test_census_pure_noise_emits_nothing() -> None:
    ad, _ = synthetic.plant_redundant_adapters(800, 4, 0.5, shared_groups=(), n_adapters=8, seed=3)
    rep = core.audit(ad)
    assert rep.effective_capacity_used == 0
    assert len(rep.dependency_census) == 0


def test_census_recall_multi_seed() -> None:
    ok = 0
    for sd in range(10):
        ad, gt = synthetic.plant_redundant_adapters(
            800, 4, 0.5, shared_groups=((0, 1, 2), (3, 4)), n_adapters=8, seed=sd
        )
        rep = core.audit(ad)
        planted = {frozenset(s) for s in gt.redundant_subsets}
        if planted.issubset(_subsets(rep)):
            ok += 1
    assert ok >= 9


def test_para_emits_no_subset() -> None:
    from adapterfax import baselines, cast

    ad, _ = synthetic.plant_redundant_adapters(
        800, 4, 0.5, shared_groups=((0, 1, 2),), n_adapters=8, seed=1
    )
    factors = {a.name: cast.to_factor(next(iter(a.layers.values()))) for a in ad}
    para = baselines.para_retained_ranks(factors)
    # PARA returns a per-adapter integer rank, never a subset of adapters.
    assert set(para.keys()) == {a.name for a in ad}
    assert all(isinstance(v, int) for v in para.values())


def test_census_scale_invariant() -> None:
    from adapterfax.model import Adapter, LoraLayer

    ad, _ = synthetic.plant_redundant_adapters(
        800, 4, 0.5, shared_groups=((0, 1, 2),), n_adapters=8, seed=2
    )
    base = _subsets(core.audit(ad))
    scaled = [
        Adapter(
            name=a.name,
            layers={
                ln: LoraLayer(
                    name=lz.name, B=lz.B * 9.1, rank=lz.rank, alpha=lz.alpha, kind=lz.kind
                )
                for ln, lz in a.layers.items()
            },
        )
        for a in ad
    ]
    assert base == _subsets(core.audit(scaled))
