#!/usr/bin/env bash
# Full probe pipeline for one model: trivia generation, correctness probe, chat responses,
# Claude style labels, style probes, prose-risk eval. Needs a GPU for 7B+ models (or use scripts/train_modal.sh).
# Usage: scripts/train_probes.sh Qwen/Qwen2.5-7B-Instruct
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROBE_MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
PY="${PYTHON:-python}"
cd "$ROOT/probes"
"$PY" gen_trivia.py --n "${N_TRIVIA:-8000}"
"$PY" train_correctness.py
"$PY" gen_responses.py --n "${N_RESPONSES:-600}"
"$PY" label_style.py
"$PY" train_style.py
