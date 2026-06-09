#!/usr/bin/env bash
# Honest-marketing / prior-art guard for adapterfax.  fail-closed (rc!=0 -> CI red).
# The guard's own regex definitions are exempt via the trailing markers.   # honest:ok
set -uo pipefail

fail=0

# --- denylist: hype / unfalsifiable / over-claim language ---------------------
DENY='state.of.the.art|\bSOTA\b|\bfirst\b|初の|fully automatic|permanent|永続|outperform|\+[0-9]+(\.[0-9]+)?\s*%|combinatorially exact|solves|resolves interference|guarantee[sd]'  # honest:ok
if grep -REn --include='*.md' --include='*.py' "$DENY" README.md docs/ src/ 2>/dev/null \
    | grep -v 'honest:ok'; then                                              # honest:ok
  echo "HONEST VIOLATION: hype/over-claim language above"
  fail=1
fi

# --- required prior-art acknowledgements -------------------------------------
for term in "WeightWatcher" "Spectrum" "erank" "PARA"; do
  if ! grep -qiF "$term" README.md; then
    echo "MISSING PRIOR-ART: $term"
    fail=1
  fi
done

# --- required NON-CLAIM disclaimers (verbatim) -------------------------------
for phrase in \
  "approximate dependency census" \
  "no calibration guarantee" \
  "no downstream accuracy"; do
  if ! grep -qF "$phrase" README.md; then
    echo "MISSING NON-CLAIM: $phrase"
    fail=1
  fi
done

# --- weights must never be tracked -------------------------------------------
if git ls-files 2>/dev/null | grep -Ei '\.(npz|npy|safetensors|pt|pth|bin|onnx|ckpt|gguf)$'; then
  echo "TRACKED WEIGHTS DETECTED"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "honesty grep: OK"
fi
exit "$fail"
