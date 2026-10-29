#!/usr/bin/env bash
# Probed model server on CPU (no GPU reservation needed).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROBE_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
exec "${PYTHON:-python}" "$ROOT/server/probed_server.py" "$@"
