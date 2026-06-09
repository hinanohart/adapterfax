# Changelog

All notable changes to adapterfax are documented here.

## [0.1.0a1] — 2026-06-09

First alpha. Data-free, CPU, post-hoc RMT audit of stacked LoRA/DoRA/(IA)³ adapters.

### Added
- `audit()` and the `adapterfax` CLI (`audit`, `gate`).
- numpy-native RMT core: Marchenko–Pastur edges, σ² estimation, MP goodness-of-fit,
  and a **parametric-null calibrated detection edge** (the pivotal quantity
  `λ_max/σ̂²`) that controls finite-sample type-I exactly — the asymptotic
  Tracy–Widom edge is kept only as an analytic cross-check.
- Robust per-adapter noise normalization to keep the stacked factor homoscedastic.
- RMT-thresholded approximate dependency census over the signal subspace
  (cross-adapter inclusion redundancy; **not a matroid**).
- Baselines run alongside for honesty: erank, PARA, cosine, TIES, DARE, Gavish–Donoho.
- Loader for PEFT and kohya/diffusers naming, DoRA magnitude, (IA)³ scales (fail-loud).
- Pre-registered sensitivity gates G1–G9 (`adapterfax gate`).

### Design decisions discovered during the build
- The detection edge operates at a **nominal 4% type-I** (one notch under the 5%
  claim) so the realized null FPR reliably stays ≤ 5% rather than straddling it.
- Per-adapter **noise** normalization (median singular value), not Frobenius:
  Frobenius normalization made signal-heavy adapters heteroscedastic and broke the
  Marchenko–Pastur assumption, biasing σ̂² and producing spurious redundant subsets.

### Not claimed
- No interference detection (sign reading is gauge-dependent; `Circuit.kind` is
  `redundant` only).
- No downstream-accuracy claim; this is a weight-space measurement instrument.
- The census is an approximate dependency census, not exact.
