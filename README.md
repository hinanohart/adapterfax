# adapterfax

[![CI](https://github.com/hinanohart/adapterfax/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/adapterfax/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10--3.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-alpha-orange)

**Data-free, CPU, post-hoc audit of a *stack* of continual-learning
LoRA / DoRA / (IA)³ adapters.** adapterfax reports an RMT-calibrated estimate of
the **effective capacity used** by the stack — how many independent signal
directions the accumulated adapters actually span above a Marchenko–Pastur noise
floor — with confidence intervals, and flags candidate **redundant subsets** of
adapters that share a direction.

It is a **measurement instrument, not an optimizer.** It reads only the adapter
weights (no data, no forward pass, no labels) and makes **no downstream accuracy
claim**.

> Status: **v0.1.0a1 (alpha).** Scope: tens of task-LoRA in a continual-learning
> lab. All headline numbers below are from synthetic experiments with planted
> ground-truth (see [Validation](#validation)).

## Why this exists

When you keep stacking task adapters during continual learning, two questions
have no cheap, data-free answer:

1. **How much of the stack is real signal vs. accumulated noise?**
2. **Which adapters are redundant with each other** (carry the same direction)?

adapterfax answers both from the weights alone, on CPU, in seconds.

## The two donor ideas (and where the novelty is)

| Donor field | What it gives | What is **not** new | What adapterfax adds |
|---|---|---|---|
| **Random Matrix Theory** (Marchenko–Pastur bulk, BBP spike transition) | a principled noise floor and a "this direction is signal" test | RMT on a **single** weight matrix is an incumbent (see below) | applies it to the **adapter-SET stack spectrum** with a calibrated, finite-sample noise edge |
| **Combinatorial dependency** (near-dependence relation over the signal subspace) | a language for "which subset is redundant" | matroid theory itself | a **cross-adapter inclusion** census that per-adapter rank pruning cannot express |

The surviving, narrow novelty is the **intersection of four properties at once**:
adapter-SET stack spectrum × MP/BBP-calibrated noise floor × SET-level redundancy
output × data-free post-hoc. Each property alone is prior art.

## Prior art (please read these)

adapterfax does **not** subsume any of these; it occupies the gap between them.

- **WeightWatcher** — RMT spectral metrics (e.g. `num_spikes`) on single weight matrices.
- **Spectrum** (arXiv:2406.06623) — per-layer MP signal-to-noise on single matrices.
- **PARA** (arXiv:2604.27796) — data-free, post-hoc global-SVD rank pruning. Reports a
  **per-adapter** retained rank; it has no notion of a cross-adapter shared subset.
- **Spectral Surgery** (arXiv:2603.03995) — training-free SVD reweighting, but data-dependent.
- **erank** / **TIES** / **DARE** — effective rank and merge baselines (all shipped alongside).
- **scikit-rmt** — RMT distribution objects (not used here; adapterfax is numpy-native).

`marchenko–pastur` on a single matrix is mature (Gavish–Donoho 2014, Johnstone 2001).
The wedge is the **adapter-set** framing plus the redundancy census, nothing more.

## Install

```bash
pip install adapterfax                 # numpy + safetensors only
pip install "adapterfax[tw]"           # + scipy, sharper Tracy–Widom cross-check
```

Core is **torch-free, CPU-only**, Python 3.10–3.13.

## Use

```bash
adapterfax audit task1.safetensors task2.safetensors task3.safetensors --json
adapterfax gate                        # run the synthetic sensitivity gates
```

```python
import adapterfax
report = adapterfax.audit(adapterfax.load_adapters(["a.safetensors", "b.safetensors"]))
print(report.effective_capacity_used, report.effective_capacity_ci)
for circuit in report.dependency_census:      # candidate redundant subsets
    print(circuit.members, circuit.margin)
```

The headline is always valid; the census is a secondary, honestly-approximate output.

## What it computes (one line of math per step)

Per layer, with factor stack `Y = [ (α/r)·B_i ]_i ∈ ℝ^{p×Σr}` (we never materialize `ΔW`):

1. robust per-adapter noise normalization (keeps the stack homoscedastic),
2. **Gram** `G = (1/p) Yᵀ Y` and `eigh(G)` (never a wide SVD),
3. noise variance `σ̂²` from the MP bulk mean,
4. a **parametric-null calibrated edge** (the pivotal quantity `λ_max/σ̂²`) controlling type-I,
5. `effective_capacity = #{ eigenvalues above the edge }` + bootstrap CI,
6. a tolerance census over the signal subspace → redundant subsets.

## Validation

All numbers come from synthetic experiments with planted ground-truth; they
characterize the instrument under known conditions and do **not** claim accuracy
improvements or fidelity on real adapter stacks.

<!-- AUTOFILLED-FROM-RESULTS -->

Pre-registered sensitivity gates (run `adapterfax gate --full`):

| Gate | Checks |
|---|---|
| G1 | supercritical spike recovery TPR ≥ 0.95 |
| G2 | null type-I FPR ≤ 0.05 |
| G3 | estimated effective rank = planted ±1 |
| G4 | estimator (bulk-mean vs median) consistency |
| G5 | determinism (byte-identical) |
| G6 | census finds cross-adapter redundancy PARA misses |
| G7 | scale invariance |
| G8 | type-I control at a second aspect ratio |
| G9 | end-to-end determinism |

## Limitations (NON-CLAIMS)

1. The census is an **approximate dependency census** (RMT-tolerance-dependent), not an
   exact one — it is *not* a matroid. The matroid axioms hold only on the RMT-denoised
   exact reconstruction, not on the tolerance relation; that exact variant is an
   experimental extra and not exposed in v0.1.
2. The MP/BBP thresholds are isotropic-iid approximations with **no calibration guarantee**
   for structured rank-r LoRA delta. They are validated on synthetic plants and a robust
   noise normalization, not proven for arbitrary real stacks.
3. Effective capacity used is a weight-space quantity with **no downstream accuracy**
   claim. A redundant subset is a candidate for inspection, not a proven safe drop.

Interference detection is **not** claimed in v0.1: a `Circuit` is `redundant` only;
sign reading is gauge-dependent and not validated. Scope is **tens** of adapters.

## License

MIT.
