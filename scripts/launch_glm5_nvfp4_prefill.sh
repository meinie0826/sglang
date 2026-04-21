#!/bin/bash
# Launch script for GLM-5 nvfp4 model in PD disaggregation PREFILL-ONLY mode.
# This node only runs prefill; decode is handled by a separate decode node.
# After prefill completes, KV cache is transferred to the decode node via mooncake.
#
# Usage:
#   MODEL_PATH=/path/to/model ./scripts/launch_glm5_nvfp4_prefill.sh
#
MODEL_PATH=${MODEL_PATH:-"/media/ssd1/glm-5-nvfp4"}
METRICS_DIR=${METRICS_DIR:-"/tmp/sglang_metrics"}

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
    --mem-fraction-static 0.78 \
    --max-running-requests 512 \
    --tp 4 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-metrics \
    --log-level-http info \
    --log-level info \
    --max-prefill-tokens 73728 \
    --attention-backend nsa \
    --nsa-prefill-backend trtllm \
    --nsa-decode-backend trtllm \
    --json-model-override-args '{"index_topk_freq": 4}' \
    --moe-runner-backend flashinfer_trtllm \
    --quantization modelopt_fp4 \
    --chunked-prefill-size 16384 \
    --max-total-tokens 196608 \
    --enable-request-time-stats-logging \
    --export-metrics-to-file \
    --export-metrics-to-file-dir "${METRICS_DIR}" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 8998 \
    "$@"
