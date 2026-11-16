#!/usr/bin/env bash
set -euo pipefail

SOURCES="pipeline_auditing/ pipelines/ tests/ main.py"

echo "==> ruff format"
python -m ruff format $SOURCES

echo "==> pylint"
python -m pylint $SOURCES
