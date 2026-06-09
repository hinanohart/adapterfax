"""S4 validation run: produce the canonical metrics JSON for v0.1.0a1.

Runs the full (non-fast) sensitivity gate suite plus a baseline-differentiation
example, env-stamps everything, and writes ``results/v0.1.0a1_metrics.json``.
All numbers are synthetic-plant ground-truth characterizations of the
instrument; no real-adapter or accuracy claim.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from adapterfax import __version__, core, gates, synthetic


def main(date_stamp: str) -> int:
    suite = gates.run_gates(fast=False)

    # baseline differentiation example: a 3-adapter shared direction that PARA
    # (per-adapter rank) cannot name but the census does.
    ad, gt = synthetic.plant_redundant_adapters(
        800, 4, 0.5, shared_groups=((0, 1, 2), (3, 4)), n_adapters=8, seed=7
    )
    rep = core.audit(ad)
    census_subsets = sorted(sorted(c.members) for c in rep.dependency_census)
    planted = sorted(sorted(s) for s in gt.redundant_subsets)

    out = {
        "version": __version__,
        "active_mode": suite["active_mode"],
        "all_gates_pass": suite["all_pass"],
        "gates": suite["gates"],
        "baseline_differentiation": {
            "planted_redundant_subsets": planted,
            "census_recovered_subsets": census_subsets,
            "census_recall_ok": all(p in census_subsets for p in planted),
            "para_retained_ranks": rep.baselines["para_retained_ranks"],
            "para_emits_subset": False,
            "erank": rep.baselines["erank"],
            "effective_capacity_used": rep.effective_capacity_used,
            "effective_capacity_ci": list(rep.effective_capacity_ci),
        },
        "env": {
            "hw": platform.machine(),
            "os": platform.system().lower(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "date": date_stamp,
            "seed": 20260609,
        },
    }
    path = Path("results/v0.1.0a1_metrics.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print(f"active_mode={out['active_mode']} all_gates_pass={out['all_gates_pass']}")
    return 0 if suite["all_pass"] else 1


if __name__ == "__main__":
    stamp = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    sys.exit(main(stamp))
