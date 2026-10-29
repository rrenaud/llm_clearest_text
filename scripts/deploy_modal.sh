#!/usr/bin/env bash
# Deploy the probed model server to Modal and record its URL for run_webui_modal.sh.
# Usage: scripts/deploy_modal.sh [model] [gpu]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODAL="${MODAL:-modal}"
KEY_FILE="$ROOT/.modal_api_key"
if [ ! -s "$KEY_FILE" ]; then
  (umask 077; openssl rand -hex 24 > "$KEY_FILE")
fi
"$MODAL" secret create --force probed-server-key PROBE_API_KEY="$(cat "$KEY_FILE")" >/dev/null
PROBE_MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}" MODAL_GPU="${2:-L4,A10G,L40S}" \
  "$MODAL" deploy "$ROOT/server/modal_app.py" | tee "$ROOT/data/modal_deploy.log"
# Redeploys don't always print the endpoint URL; it is stable, so keep the previous one then.
URL="$(grep -oE 'https://[^ ]+\.modal\.run' "$ROOT/data/modal_deploy.log" | tail -1 || true)"
if [ -n "$URL" ]; then echo "$URL" > "$ROOT/.modal_url"; fi
echo "URL: $(cat "$ROOT/.modal_url")"
