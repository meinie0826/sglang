#!/bin/bash
# Launch script for GLM-5 nvfp4 model (open-source sglang compatible)
# Removed private-only args: --ttft-norm-base, --opt-model-name,
#   --skip-tool-call-schema-check, --batch-policy, --max-input-length, --max-output-length
# Changed: --tool-call-parser glm47 -> glm (glm47 is deprecated alias)
# Changed: --max-running-request -> --max-running-requests
# TTFT data collection enabled via --export-metrics-to-file

MODEL_PATH=${MODEL_PATH:-"/media/ssd1/glm-5-nvfp4"}
METRICS_DIR=${METRICS_DIR:-"/tmp/sglang_metrics"}

mkdir -p "${METRICS_DIR}"

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --enable-cache-report \
    --tool-call-parser glm \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.78 \
    --max-running-requests 512 \
    --tokenizer-worker-num 2 \
    --cuda-graph-bs 1 16 32 64 \
    --tp 4 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-metrics \
    --log-level-http info \
    --log-level info \
    --load-balance-method round_robin \
    --max-prefill-tokens 73728 \
    --hicache-write-policy write_back \
    --enable-hierarchical-cache \
    --hicache-size 200 \
    --hicache-io-backend kernel \
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
    "$@"
