#!/usr/bin/env bash
# Clone Kronos, create a Python 3.12 venv (with uv when available), and install
# Kronos's requirements plus yfinance.
set -euo pipefail
cd "$(dirname "$0")"

[ -d Kronos ] || git clone https://github.com/shiyu-coder/Kronos.git

if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r Kronos/requirements.txt yfinance
else
    python3.12 -m venv .venv
    .venv/bin/pip install -r Kronos/requirements.txt yfinance
fi
