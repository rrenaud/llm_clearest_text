#!/usr/bin/env bash
# Open WebUI (pip backend) serving the probe-rendering frontend fork, pointed at the probed model server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export FRONTEND_BUILD_DIR="$ROOT/vendor/open-webui/build"
export DATA_DIR="$ROOT/webui-data"
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-http://127.0.0.1:8100/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export ENABLE_OLLAMA_API=False
export ENABLE_PERSISTENT_CONFIG=False   # env vars (model server URL/key) win over settings saved in the DB
export DEFAULT_MODELS="${DEFAULT_MODELS:-qwen2.5-7b-instruct-probed}"
export WEBUI_AUTH="${WEBUI_AUTH:-False}"
export ENABLE_TITLE_GENERATION=False ENABLE_TAGS_GENERATION=False ENABLE_FOLLOW_UP_GENERATION=False ENABLE_AUTOCOMPLETE_GENERATION=False
export RAG_EMBEDDING_MODEL_AUTO_UPDATE=False OFFLINE_MODE=True
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""   # the UI itself must not touch the GPU
mkdir -p "$DATA_DIR"
exec "${OPEN_WEBUI:-$ROOT/envs/webui/bin/open-webui}" serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
