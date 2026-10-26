#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/artifacts/voice-demo-runtime/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "Missing runtime: $PYTHON"
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" -B scripts/demo_dashboard.py
