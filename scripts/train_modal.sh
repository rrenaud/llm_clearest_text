#!/usr/bin/env bash
# Train probes for a model on Modal, then copy the small probe files into data/<slug>/ for deploy_modal.sh.
# Usage: scripts/train_modal.sh [model] [gpu]      (rerun with FETCH_ONLY=1 to just download results)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODAL="${MODAL:-modal}"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"; SLUG="${MODEL##*/}"
if [ -z "${FETCH_ONLY:-}" ]; then
  "$MODAL" secret create --force anthropic-key ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(cat "$HOME/anthropic_api_key.txt")}" >/dev/null
  PROBE_MODEL="$MODEL" MODAL_GPU="${2:-L40S}" "$MODAL" run "$ROOT/server/modal_train.py"
fi
mkdir -p "$ROOT/data/$SLUG"
for f in probe_correctness.npz probe_correctness.json probe_boilerplate.npz probe_filler.npz probe_style.json eval_prose_risk.json; do
  "$MODAL" volume get --force probed-data "$SLUG/$f" "$ROOT/data/$SLUG/$f"
done
echo "probes in $ROOT/data/$SLUG"
