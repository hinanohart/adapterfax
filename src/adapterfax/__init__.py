"""adapterfax — data-free, CPU, post-hoc RMT audit of stacked LoRA/DoRA/(IA)3 adapters.

A measurement instrument, not an optimizer.  Reports an RMT-calibrated estimate of
the effective capacity used by a stack of adapters, with confidence intervals, and
flags candidate redundant subsets relative to a Marchenko-Pastur noise floor.
Makes no downstream-accuracy claims.
"""

from __future__ import annotations

__version__ = "0.1.0a1"

__all__ = ["__version__"]
