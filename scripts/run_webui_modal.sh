#!/usr/bin/env bash
# Local Open WebUI pointed at the Modal-hosted probed server (deploy first with scripts/deploy_modal.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export OPENAI_API_BASE_URL="$(cat "$ROOT/.modal_url")/v1"
export OPENAI_API_KEY="$(cat "$ROOT/.modal_api_key")"
export PORT="${PORT:-8502}"   # forwarded to the laptop
exec "$ROOT/scripts/run_webui.sh"
