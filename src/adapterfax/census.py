"""RMT-thresholded approximate dependency census over a near-dependence relation.

**This is NOT a matroid.**  The tolerance-based near-dependence relation does
not satisfy the exchange/augmentation axioms, so the output is an *independence
system*, not a matroid.  A true linear matroid would hold only on the
RMT-denoised exact low-rank reconstruction, which is an experimental extra and
not exposed in v0.1.

The census operates entirely in the *signal subspace* surviving the
Marchenko–Pastur noise floor: for each signal direction (a Gram eigenvector
with super-edge eigenvalue) it identifies the set of adapters that jointly load
on it.  A direction carried by ≥2 adapters is a **cross-adapter inclusion
redundancy** — the kind per-adapter rank pruning (PARA) cannot see, because each
adapter is individually full-rank.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import FloatArray

__all__ = ["Circuit", "dependency_census"]


@dataclass(frozen=True)
class Circuit:
    """An approximately redundant subset of adapters sharing a signal direction.

    ``kind`` is ``'redundant'`` in v0.1.  ``margin`` is how far each member's
    energy fraction sits above the tolerance (larger ⇒ more robust).  The
    direction index ``direction`` ties the circuit to a Gram eigenvector.
    """

    members: tuple[str, ...]
    kind: str
    margin: float
    direction: int
    energy_fractions: tuple[float, ...]


def _adapter_blocks(ownership: list[tuple[str, int, int]]) -> dict[str, list[tuple[int, int]]]:
    blocks: dict[str, list[tuple[int, int]]] = {}
    for name, c0, c1 in ownership:
        blocks.setdefault(name, []).append((c0, c1))
    return blocks


def dependency_census(
    signal_vecs: FloatArray,
    ownership: list[tuple[str, int, int]],
    tol: float,
) -> list[Circuit]:
    """Find cross-adapter inclusion redundancies in the signal subspace.

    Parameters
    ----------
    signal_vecs:
        ``m × k`` matrix whose columns are the Gram eigenvectors classified as
        signal (super-edge).  Column rows are indexed like the stacked factor
        columns, owned per :paramref:`ownership`.
    ownership:
        ``[(adapter_name, col_start, col_end), ...]`` over the ``m`` columns.
    tol:
        An adapter participates in a direction iff its share of that
        direction's squared energy exceeds ``tol`` (default supplied by the
        caller from the RMT scale).  ``≥2`` participants ⇒ redundant subset.
    """
    if signal_vecs.ndim != 2 or signal_vecs.shape[1] == 0:
        return []
    blocks = _adapter_blocks(ownership)
    names = list(blocks.keys())
    circuits: list[Circuit] = []
    for j in range(signal_vecs.shape[1]):
        v = signal_vecs[:, j]
        total = float(v @ v)
        if total <= 0.0:
            continue
        fracs: dict[str, float] = {}
        for name in names:
            energy = 0.0
            for c0, c1 in blocks[name]:
                seg = v[c0:c1]
                energy += float(seg @ seg)
            fracs[name] = energy / total
        participants = [(nm, fr) for nm, fr in fracs.items() if fr > tol]
        if len(participants) >= 2:
            participants.sort(key=lambda t: -t[1])
            members = tuple(nm for nm, _ in participants)
            efr = tuple(round(fr, 6) for _, fr in participants)
            margin = float(min(fr for _, fr in participants) - tol)
            circuits.append(
                Circuit(
                    members=members,
                    kind="redundant",
                    margin=margin,
                    direction=j,
                    energy_fractions=efr,
                )
            )
    return _merge_identical(circuits)


def _merge_identical(circuits: list[Circuit]) -> list[Circuit]:
    """Collapse circuits over identical member sets (keep the largest margin)."""
    best: dict[frozenset[str], Circuit] = {}
    for c in circuits:
        key = frozenset(c.members)
        cur = best.get(key)
        if cur is None or c.margin > cur.margin:
            best[key] = c
    return sorted(best.values(), key=lambda c: (-len(c.members), -c.margin))


def default_tolerance(n_adapters: int) -> float:
    """Default energy-fraction tolerance: a fraction of an even split.

    With ``N`` adapters an even split puts ``1/N`` of a shared direction's
    energy on each; we flag participation above ``0.5/N`` (half an even share),
    which cleanly separates true sharers (~``1/g``) from leakage (≈0).
    """
    return 0.5 / max(2, n_adapters)
