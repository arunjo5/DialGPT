#!/usr/bin/env bash
set -euo pipefail

# Run the project using the venv Python to avoid interpreter mismatch when using --reload.
# Usage: ./run.sh

if [ -x ".venv/bin/python" ]; then
  echo "Using virtualenv python: .venv/bin/python"
  exec .venv/bin/python -m uvicorn main:app --reload
else
  echo "Virtualenv not found at .venv. Create it with (recommended):"
  echo "  python3.11 -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt"
  exit 1
fi

