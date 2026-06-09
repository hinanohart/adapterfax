"""Frozen, JSON-serializable Report dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ._noclaim import NON_CLAIMS
from .census import Circuit

SCHEMA_VERSION = 1

# The primary headline quantity is always valid; secondary outputs may degrade.
HEADLINE = (
    "RMT-calibrated estimate of the effective capacity used by a stack of "
    "LoRA/DoRA/(IA)3 adapters, data-free and on CPU, with confidence intervals. "
    "A measurement instrument, not an optimizer; makes no accuracy claims."
)


@dataclass(frozen=True)
class Direction:
    layer: str
    eigenvalue: float
    classification: str  # "signal" | "bulk_noise"
    loading_adapters: tuple[str, ...]


@dataclass(frozen=True)
class LayerReport:
    layer: str
    p: int
    m: int
    gamma: float
    sigma2_hat: float
    mp_edge: float
    detect_threshold: float
    effective_capacity: int
    ks_stat: float
    ks_pvalue: float
    n_circuits: int


@dataclass(frozen=True)
class Calibration:
    gamma_mean: float
    sigma2_hat_mean: float
    estimator: str
    edge_method: str
    tw_analytic_crosscheck: bool
    mp_goodness_of_fit_min_p: float
    degraded_to: str | None
    disclaimer: str


@dataclass(frozen=True)
class Report:
    schema_version: int
    active_mode: str
    headline: str
    effective_capacity_used: int
    effective_capacity_per_layer: dict[str, int]
    effective_capacity_ci: tuple[float, float]
    rank_certificate: tuple[Direction, ...]
    dependency_census: tuple[Circuit, ...]
    baselines: dict[str, object]
    per_layer: tuple[LayerReport, ...]
    calibration: Calibration
    provenance: dict[str, object]
    non_claims: tuple[str, str, str] = field(default=NON_CLAIMS)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        # asdict already recursed through nested dataclasses and tuples.
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=_fallback)


def _fallback(obj: object) -> object:
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")
