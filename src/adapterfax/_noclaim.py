"""Frozen NON-CLAIM constants and disclaimers (single source, grepped by CI)."""

from __future__ import annotations

# The three load-bearing honesty statements.  Reproduced verbatim in the README;
# CI greps the README for each.  Do not paraphrase.
NON_CLAIMS: tuple[str, str, str] = (
    "the census is an approximate dependency census (RMT-tolerance-dependent), not an exact one",
    "MP/BBP thresholds are isotropic-iid approximations with no calibration guarantee "
    "for structured rank-r LoRA delta",
    "effective capacity used is a weight-space quantity; no downstream accuracy claim",
)

# Injected immediately before any block of synthetic-only numbers.
SYNTHETIC_DISCLAIMER: str = (
    "Numbers below come from synthetic experiments with planted ground-truth. "
    "They characterize the instrument under known conditions and do not claim "
    "accuracy improvements, interference resolution, or fidelity on real adapter stacks."
)

# Required prior-art acknowledgements (subset greppable in README).
PRIOR_ART: tuple[str, ...] = (
    "WeightWatcher",
    "Spectrum (arXiv:2406.06623)",
    "erank",
    "PARA (arXiv:2604.27796)",
    "Spectral Surgery (arXiv:2603.03995)",
    "TIES",
    "DARE",
    "scikit-rmt",
)
