#!/bin/bash
# Launch script for GLM-5 nvfp4 model (open-source sglang compatible)
# Changed: --tool-call-parser glm47 -> glm (glm47 is deprecated alias)
# Removed to reduce startup issues:
#   --speculative-algorithm EAGLE (and related spec args) - complex warmup, easy to hang
#   --enable-hierarchical-cache / --hicache-* - extra IO backend complexity
#   --load-balance-method round_robin - irrelevant for single instance
#   --tokenizer-worker-num 2 - default is fine

MODEL_PATH=${MODEL_PATH:-"/media/ssd1/glm-5-nvfp4"}
METRICS_DIR=${METRICS_DIR:-"/tmp/sglang_metrics"}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-"0.78"}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-"512"}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-"1 16 32 64"}
TP_SIZE=${TP_SIZE:-"4"}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-"73728"}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-"16384"}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-"196608"}

mkdir -p "${METRICS_DIR}"

# Clean up stale/broken symlinks only (not the whole dir), to avoid re-downloading cubins.
find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins/ \
    -maxdepth 3 -type l ! -e 2>/dev/null | xargs rm -f

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --enable-cache-report \
    --tool-call-parser glm \
    --reasoning-parser glm45 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --cuda-graph-bs ${CUDA_GRAPH_BS} \
    --tp "${TP_SIZE}" \
    --kv-cache-dtype fp8_e4m3 \
    --enable-metrics \
    --log-level-http info \
    --log-level info \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --attention-backend nsa \
    --nsa-prefill-backend trtllm \
    --nsa-decode-backend trtllm \
    --json-model-override-args '{"index_topk_freq": 4}' \
    --moe-runner-backend flashinfer_trtllm \
    --quantization modelopt_fp4 \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --enable-request-time-stats-logging \
    --export-metrics-to-file \
    --export-metrics-to-file-dir "${METRICS_DIR}" \
    "$@"
