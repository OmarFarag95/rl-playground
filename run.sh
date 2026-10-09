#!/usr/bin/env bash
# Start Splash Lab. Creates a virtual environment on first run (reusing an installed PyTorch).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -r requirements.txt
fi
cd backend
exec ../.venv/bin/python -m rlplay "$@"
