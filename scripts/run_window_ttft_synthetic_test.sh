#!/bin/bash
set -euo pipefail

# One-command synthetic validation for window-level p50 TTFT prediction.
#
# Usage:
#   bash scripts/run_window_ttft_synthetic_test.sh
#   WINDOW_SECONDS=120 DURATION_SECONDS=14400 bash scripts/run_window_ttft_synthetic_test.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/media/ssd1/glm-5-nvfp4}"
WINDOW_SECONDS="${WINDOW_SECONDS:-60}"
DURATION_SECONDS="${DURATION_SECONDS:-21600}"
MIN_WINDOW_REQUESTS="${MIN_WINDOW_REQUESTS:-5}"
SEED="${SEED:-20260421}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/tmp/window_ttft_synth.jsonl}"

echo "=== Window TTFT Synthetic Test ==="
echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "WINDOW_SECONDS=${WINDOW_SECONDS}"
echo "DURATION_SECONDS=${DURATION_SECONDS}"
echo "MIN_WINDOW_REQUESTS=${MIN_WINDOW_REQUESTS}"
echo "SEED=${SEED}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo

python3 "${ROOT_DIR}/scripts/test_window_ttft_synthetic.py" \
  --seed "${SEED}" \
  --window-seconds "${WINDOW_SECONDS}" \
  --duration-seconds "${DURATION_SECONDS}" \
  --min-window-requests "${MIN_WINDOW_REQUESTS}" \
  --dump-jsonl "${OUTPUT_JSONL}"
