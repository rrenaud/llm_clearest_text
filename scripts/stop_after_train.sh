#!/usr/bin/env bash
# Wait for the training launcher to exit (or MAX_HOURS), then make sure no Modal GPU is left running.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODAL="${MODAL:-modal}"
deadline=$(( $(date +%s) + ${MAX_HOURS:-4} * 3600 ))
while pgrep -u "$USER" -f "train_moda[l].sh" >/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do sleep 60; done
for app in probe-train probed-chat-server; do "$MODAL" app stop -y "$app" >/dev/null 2>&1 || true; done
sleep 30
echo "$(date) stopped; active containers: $("$MODAL" container list 2>&1 | grep -c 'ta-')"
