#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/media/ssd1/glm-5-nvfp4}"
SERVER_LAUNCH_SCRIPT="${SERVER_LAUNCH_SCRIPT:-${ROOT_DIR}/scripts/launch_glm5_nvfp4.sh}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"

WINDOW_SECONDS="${WINDOW_SECONDS:-60}"
TRAINING_WINDOW_SECONDS="${TRAINING_WINDOW_SECONDS:-900}"
RETRAIN_INTERVAL_SECONDS="${RETRAIN_INTERVAL_SECONDS:-30}"

REQUEST_RATE="${REQUEST_RATE:-2}"
TEST_DURATION_SECONDS="${TEST_DURATION_SECONDS:-180}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-4096}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.3}"
SEED="${SEED:-20260421}"

START_SERVER="${START_SERVER:-1}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"
SERVER_STARTUP_GRACE_SECONDS="${SERVER_STARTUP_GRACE_SECONDS:-10}"

RUN_TAG="${RUN_TAG:-window_ttft_online_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/${RUN_TAG}}"
METRICS_DIR="${METRICS_DIR:-${ARTIFACT_DIR}/metrics}"
SERVER_LOG="${SERVER_LOG:-${ARTIFACT_DIR}/server.log}"
CLIENT_LOG="${CLIENT_LOG:-${ARTIFACT_DIR}/client.log}"
CLIENT_OUTPUT_JSONL="${CLIENT_OUTPUT_JSONL:-${ARTIFACT_DIR}/bench_output.jsonl}"

EVENT_LOG_PATH="${EVENT_LOG_PATH:-/tmp/sglang_window_ttft_events.jsonl}"
MODEL_JSON_PATH="${MODEL_JSON_PATH:-/tmp/sglang_window_ttft_model.json}"

mkdir -p "${ARTIFACT_DIR}" "${METRICS_DIR}"

if [[ -z "${NUM_PROMPTS}" ]]; then
  NUM_PROMPTS="$(python3 - <<PY
import math
request_rate = float("${REQUEST_RATE}")
duration = float("${TEST_DURATION_SECONDS}")
print(max(1, int(math.ceil(request_rate * duration))))
PY
)"
fi

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo
    echo "[cleanup] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_endpoint() {
  local url="$1"
  local timeout="$2"
  local process_pid="${3:-}"
  local waited=0
  while true; do
    if curl -fsS -H "Authorization: Bearer None" "${url}" >/dev/null 2>&1; then
      return 0
    fi

    if [[ -n "${process_pid}" ]] && ! kill -0 "${process_pid}" >/dev/null 2>&1; then
      echo "[error] process exited early while waiting for ${url}" >&2
      return 1
    fi

    if (( waited >= timeout )); then
      echo "[error] endpoint ${url} not ready after ${timeout}s" >&2
      return 1
    fi

    sleep 1
    waited=$((waited + 1))
  done
}

summarize_predictor_events() {
  python3 - <<PY
import json
from pathlib import Path

path = Path("${EVENT_LOG_PATH}")
if not path.exists():
    print("[summary] predictor event log not found:", path)
    raise SystemExit(0)

snapshots = []
retrains = []
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if obj.get("event") == "window_snapshot":
        snapshots.append(obj)
    elif obj.get("event") == "retrain":
        retrains.append(obj)

print(f"[summary] event_log={path}")
print(f"[summary] window_snapshots={len(snapshots)} retrains={len(retrains)}")
if snapshots:
    last = snapshots[-1]
    print(
        "[summary] last_snapshot "
        f"window_seconds={last.get('window_ttft_window_seconds')} "
        f"actual_p50_ms={last.get('window_p50_ttft_ms'):.3f} "
        f"predicted_p50_ms={last.get('predicted_window_p50_ttft_ms'):.3f} "
        f"abs_error_ms={last.get('window_ttft_abs_error_ms'):.3f} "
        f"arrival_qps={last.get('arrival_qps_60s'):.3f} "
        f"finished_qps={last.get('finished_qps_60s'):.3f} "
        f"model_ready={last.get('window_ttft_predictor_ready')} "
        f"model_version={last.get('window_ttft_model_version')}"
    )
if retrains:
    last = retrains[-1]
    print(
        "[summary] last_retrain "
        f"model_version={last.get('model_version')} "
        f"sample_count={last.get('sample_count')} "
        f"mae_ms={last.get('mae_ms'):.3f}"
    )
PY
}

echo "=== Window TTFT Online Smoke Test ==="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVER_LAUNCH_SCRIPT=${SERVER_LAUNCH_SCRIPT}"
echo "BASE_URL=${BASE_URL}"
echo "WINDOW_SECONDS=${WINDOW_SECONDS}"
echo "TRAINING_WINDOW_SECONDS=${TRAINING_WINDOW_SECONDS}"
echo "RETRAIN_INTERVAL_SECONDS=${RETRAIN_INTERVAL_SECONDS}"
echo "REQUEST_RATE=${REQUEST_RATE}"
echo "TEST_DURATION_SECONDS=${TEST_DURATION_SECONDS}"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
echo "RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN}"
echo "RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN}"
echo "RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}"
echo "SEED=${SEED}"
echo "ARTIFACT_DIR=${ARTIFACT_DIR}"
echo "SERVER_LOG=${SERVER_LOG}"
echo "CLIENT_LOG=${CLIENT_LOG}"
echo "CLIENT_OUTPUT_JSONL=${CLIENT_OUTPUT_JSONL}"
echo "EVENT_LOG_PATH=${EVENT_LOG_PATH}"
echo "MODEL_JSON_PATH=${MODEL_JSON_PATH}"
echo

rm -f "${CLIENT_OUTPUT_JSONL}" "${CLIENT_LOG}" "${SERVER_LOG}" "${EVENT_LOG_PATH}" "${MODEL_JSON_PATH}"

if [[ "${START_SERVER}" == "1" ]]; then
  echo "[1/3] starting server"
  (
    cd "${ROOT_DIR}"
    export MODEL_PATH
    export METRICS_DIR
    export SGLANG_WINDOW_TTFT_PREDICTOR_WINDOW_SECONDS="${WINDOW_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_TRAINING_WINDOW_SECONDS="${TRAINING_WINDOW_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_RETRAIN_INTERVAL_SECONDS="${RETRAIN_INTERVAL_SECONDS}"
    exec bash "${SERVER_LAUNCH_SCRIPT}" --host "${HOST}" --port "${PORT}"
  ) >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  sleep "${SERVER_STARTUP_GRACE_SECONDS}"
  wait_for_endpoint "${BASE_URL}/v1/models" "${READY_TIMEOUT_SECONDS}" "${SERVER_PID}"
  echo "[1/3] server ready pid=${SERVER_PID}"
else
  echo "[1/3] reuse existing server at ${BASE_URL}"
fi

echo "[2/3] running random chat client"
(
  cd "${ROOT_DIR}"
  python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name random \
    --base-url "${BASE_URL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --random-input-len "${RANDOM_INPUT_LEN}" \
    --random-output-len "${RANDOM_OUTPUT_LEN}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --disable-stream \
    --apply-chat-template \
    --output-file "${CLIENT_OUTPUT_JSONL}" \
    --output-details \
    --seed "${SEED}" \
    --ready-check-timeout-sec 0
) 2>&1 | tee "${CLIENT_LOG}"

echo "[3/3] summarizing predictor artifacts"
summarize_predictor_events

echo
echo "Finished. Artifacts:"
echo "  server log: ${SERVER_LOG}"
echo "  client log: ${CLIENT_LOG}"
echo "  bench jsonl: ${CLIENT_OUTPUT_JSONL}"
echo "  metrics dir: ${METRICS_DIR}"
echo "  predictor event log: ${EVENT_LOG_PATH}"
echo "  predictor model: ${MODEL_JSON_PATH}"
