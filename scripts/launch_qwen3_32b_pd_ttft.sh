#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/sgl-model-gateway/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/media/ssd1/qwen3-32b}"

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-31000}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
DECODE_PORT="${DECODE_PORT:-31001}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

PREFILL_CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-0,1}"
DECODE_CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-2,3}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-2}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-2}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
DISAGG_TRANSFER_BACKEND="${DISAGG_TRANSFER_BACKEND:-mooncake}"
DISAGG_IB_DEVICE="${DISAGG_IB_DEVICE:-}"

WINDOW_SECONDS="${WINDOW_SECONDS:-60}"
TRAINING_WINDOW_SECONDS="${TRAINING_WINDOW_SECONDS:-900}"
RETRAIN_INTERVAL_SECONDS="${RETRAIN_INTERVAL_SECONDS:-30}"
FUTURE_QPS_VALUES="${FUTURE_QPS_VALUES:-0.15,0.3,0.45,0.6,0.8,1.1,1.4}"
FUTURE_QPS_INTERVAL_SECONDS="${FUTURE_QPS_INTERVAL_SECONDS:-10}"
FUTURE_QPS_HORIZON_SECONDS="${FUTURE_QPS_HORIZON_SECONDS:-15}"
FUTURE_QPS_MATCH_TOLERANCE="${FUTURE_QPS_MATCH_TOLERANCE:-0.1}"

RUN_TAG="${RUN_TAG:-qwen3_32b_pd_ttft_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_BASE_DIR="${ARTIFACT_BASE_DIR:-${ROOT_DIR}/artifacts/ttft_online}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ARTIFACT_BASE_DIR}/${RUN_TAG}}"
PREFILL_DIR="${ARTIFACT_DIR}/prefill"
DECODE_DIR="${ARTIFACT_DIR}/decode"
ROUTER_DIR="${ARTIFACT_DIR}/router"
PREFILL_METRICS_DIR="${PREFILL_DIR}/metrics"
DECODE_METRICS_DIR="${DECODE_DIR}/metrics"
PREDICTOR_DIR="${DECODE_DIR}/predictor"
MODEL_JSON_PATH="${PREDICTOR_DIR}/sglang_window_ttft_model.json"
EVENT_LOG_PATH="${PREDICTOR_DIR}/sglang_window_ttft_events.jsonl"
FUTURE_QPS_LOG_PATH="${PREDICTOR_DIR}/sglang_window_ttft_future_qps.jsonl"
PREFILL_LOG_PATH="${PREFILL_DIR}/server.log"
DECODE_LOG_PATH="${DECODE_DIR}/server.log"
ROUTER_LOG_PATH="${ROUTER_DIR}/router.log"

mkdir -p \
    "${PREFILL_METRICS_DIR}" \
    "${DECODE_METRICS_DIR}" \
    "${PREDICTOR_DIR}" \
    "${ROUTER_DIR}"

cleanup() {
    local exit_code=$?
    if [[ -n "${ROUTER_PID:-}" ]]; then
        kill "${ROUTER_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${DECODE_PID:-}" ]]; then
        kill "${DECODE_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${PREFILL_PID:-}" ]]; then
        kill "${PREFILL_PID}" >/dev/null 2>&1 || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

wait_for_http() {
    local name="$1"
    local url="$2"
    local log_path="$3"
    local check_count=0
    echo "[wait] ${name}: ${url}"
    while true; do
        if curl -s -f "${url}" >/dev/null 2>&1; then
            echo "[ready] ${name}: ${url}"
            return 0
        fi
        sleep 5
        check_count=$((check_count + 1))
        if (( check_count % 6 == 0 )); then
            echo "[wait] ${name} still not ready, tailing ${log_path}"
            tail -n 20 "${log_path}" 2>/dev/null || true
        fi
    done
}

python - <<'PY'
import importlib
import sys

for module in ("xgboost", "sglang_router.launch_router"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        print(f"[error] missing python dependency: {module}: {exc}", file=sys.stderr)
        sys.exit(1)
PY

DISAGG_IB_ARGS=()
if [[ -n "${DISAGG_IB_DEVICE}" ]]; then
    DISAGG_IB_ARGS+=(--disaggregation-ib-device "${DISAGG_IB_DEVICE}")
fi

COMMON_SERVER_ARGS=(
    --model-path "${MODEL_PATH}"
    --trust-remote-code
    --host
)

COMMON_RUNTIME_ARGS=(
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --enable-metrics
    --log-level-http info
    --log-level info
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --enable-request-time-stats-logging
    --export-metrics-to-file
)

echo "=== Qwen3-32B PD TTFT Launch ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "ARTIFACT_DIR=${ARTIFACT_DIR}"
echo "PREFILL=${PREFILL_HOST}:${PREFILL_PORT} GPUs=${PREFILL_CUDA_VISIBLE_DEVICES} TP=${PREFILL_TP_SIZE}"
echo "DECODE=${DECODE_HOST}:${DECODE_PORT} GPUs=${DECODE_CUDA_VISIBLE_DEVICES} TP=${DECODE_TP_SIZE}"
echo "ROUTER=${ROUTER_HOST}:${ROUTER_PORT}"
echo "WINDOW_SECONDS=${WINDOW_SECONDS}"
echo "FUTURE_QPS_VALUES=${FUTURE_QPS_VALUES}"
echo "FUTURE_QPS_INTERVAL_SECONDS=${FUTURE_QPS_INTERVAL_SECONDS}"
echo "FUTURE_QPS_HORIZON_SECONDS=${FUTURE_QPS_HORIZON_SECONDS}"
echo "FUTURE_QPS_MATCH_TOLERANCE=${FUTURE_QPS_MATCH_TOLERANCE}"
echo

(
    export CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES}"
    export MODEL_PATH
    exec python -m sglang.launch_server \
        "${COMMON_SERVER_ARGS[@]}" "${PREFILL_HOST}" \
        --port "${PREFILL_PORT}" \
        "${COMMON_RUNTIME_ARGS[@]}" \
        --export-metrics-to-file-dir "${PREFILL_METRICS_DIR}" \
        --tp "${PREFILL_TP_SIZE}" \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend "${DISAGG_TRANSFER_BACKEND}" \
        "${DISAGG_IB_ARGS[@]}" \
        --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
        >"${PREFILL_LOG_PATH}" 2>&1
) &
PREFILL_PID=$!
echo "[start] prefill pid=${PREFILL_PID} log=${PREFILL_LOG_PATH}"

(
    export CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES}"
    export MODEL_PATH
    export SGLANG_WINDOW_TTFT_PREDICTOR_WINDOW_SECONDS="${WINDOW_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_TRAINING_WINDOW_SECONDS="${TRAINING_WINDOW_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_RETRAIN_INTERVAL_SECONDS="${RETRAIN_INTERVAL_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_MODEL_PATH="${MODEL_JSON_PATH}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_EVENT_LOG_PATH="${EVENT_LOG_PATH}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_VALUES="${FUTURE_QPS_VALUES}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_INTERVAL_SECONDS="${FUTURE_QPS_INTERVAL_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_LOG_PATH="${FUTURE_QPS_LOG_PATH}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_HORIZON_SECONDS="${FUTURE_QPS_HORIZON_SECONDS}"
    export SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_MATCH_TOLERANCE="${FUTURE_QPS_MATCH_TOLERANCE}"
    exec python -m sglang.launch_server \
        "${COMMON_SERVER_ARGS[@]}" "${DECODE_HOST}" \
        --port "${DECODE_PORT}" \
        "${COMMON_RUNTIME_ARGS[@]}" \
        --export-metrics-to-file-dir "${DECODE_METRICS_DIR}" \
        --tp "${DECODE_TP_SIZE}" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend "${DISAGG_TRANSFER_BACKEND}" \
        "${DISAGG_IB_ARGS[@]}" \
        --base-gpu-id 0 \
        >"${DECODE_LOG_PATH}" 2>&1
) &
DECODE_PID=$!
echo "[start] decode pid=${DECODE_PID} log=${DECODE_LOG_PATH}"

wait_for_http "prefill" "http://${PREFILL_HOST}:${PREFILL_PORT}/health" "${PREFILL_LOG_PATH}"
wait_for_http "decode" "http://${DECODE_HOST}:${DECODE_PORT}/health" "${DECODE_LOG_PATH}"

(
    exec python -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" \
        --decode "http://${DECODE_HOST}:${DECODE_PORT}" \
        --host "${ROUTER_HOST}" \
        --port "${ROUTER_PORT}" \
        >"${ROUTER_LOG_PATH}" 2>&1
) &
ROUTER_PID=$!
echo "[start] router pid=${ROUTER_PID} log=${ROUTER_LOG_PATH}"

wait_for_http "router" "http://127.0.0.1:${ROUTER_PORT}/health" "${ROUTER_LOG_PATH}"

echo
echo "=== Ready ==="
echo "Router endpoint:"
echo "  http://127.0.0.1:${ROUTER_PORT}"
echo "Decode predictor status:"
echo "  curl http://${DECODE_HOST}:${DECODE_PORT}/window_ttft_predictor_status"
echo "Decode what-if qps:"
echo "  curl -X POST http://${DECODE_HOST}:${DECODE_PORT}/predict_window_ttft -H 'Content-Type: application/json' -d '{\"future_qps\": 2.0}'"
echo "Artifacts:"
echo "  ${ARTIFACT_DIR}"
echo

wait
