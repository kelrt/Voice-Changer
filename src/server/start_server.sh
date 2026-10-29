#!/bin/bash
# Run from anywhere: resolve the repo root relative to this script.
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate
python src/server/server.py
