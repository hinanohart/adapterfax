<!-- WIP scaffold. Numbers are filled in at S4/S5 from results/ only. -->
# adapterfax

> **Status: v0.1.0a1 (alpha).** Work in progress — this README is finalized after
> the validation run; numeric claims come only from `results/`.

A **data-free, CPU, post-hoc** measurement instrument for stacks of continual-learning
**LoRA / DoRA / (IA)³** adapters. It reports an RMT-calibrated estimate of the
**effective capacity used** by the stack, with confidence intervals, and flags
candidate redundant subsets relative to a Marchenko–Pastur noise floor.

It is a **measurement instrument, not an optimizer**, and makes **no downstream
accuracy claims**.

License: MIT.
