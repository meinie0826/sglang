"""
Experiment 1: Pure Prefill Baseline
====================================
Send single requests with max_new_tokens=1 at zero load.
Measures pure prefill latency as a function of prompt length.
No concurrent requests, so there is no decode interference.

Usage:
    python scripts/exp1_pure_prefill.py --host 127.0.0.1 --port 30000
"""

import argparse
import json
import time

import numpy as np
import requests

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536]
REPEATS = 5  # Number of requests per seq_len for averaging


def send_request(host: str, port: int, prompt_len: int, max_new_tokens: int = 1) -> float:
    """Send a single request and return TTFT in seconds."""
    # Use random integer token IDs to avoid prefix cache hits across requests.
    import random
    input_ids = [random.randint(100, 50000) for _ in range(prompt_len)]
    url = f"http://{host}:{port}/generate"
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
        },
        "stream": False,
        "log_metrics": True,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=600)
    t1 = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()

    # Use server-reported ttft if available, else fall back to client-side e2e
    meta = data.get("meta_info", {})
    ttft = meta.get("ttft")
    if ttft is None:
        ttft = t1 - t0  # fallback: client-side (includes network)
    return ttft, meta.get("prompt_tokens", prompt_len), meta.get("cached_tokens", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", default="exp1_pure_prefill.json")
    args = parser.parse_args()

    results = []
    print(f"{'seq_len':>10} {'prompt_tok':>12} {'cached_tok':>12} {'extend_tok':>12} "
          f"{'ttft_mean(ms)':>14} {'ttft_p50(ms)':>13} {'ttft_p95(ms)':>13}")
    print("-" * 100)

    for seq_len in SEQ_LENS:
        ttfts = []
        prompt_tokens = 0
        cached_tokens = 0
        for i in range(args.repeats):
            try:
                ttft, pt, ct = send_request(args.host, args.port, seq_len)
                ttfts.append(ttft)
                prompt_tokens = pt
                cached_tokens = ct
                time.sleep(0.5)  # Brief pause between requests to keep system idle
            except Exception as e:
                print(f"  WARNING: request failed for seq_len={seq_len}: {e}")

        if not ttfts:
            continue

        extend_tokens = max(0, prompt_tokens - cached_tokens)
        mean_ms = np.mean(ttfts) * 1000
        p50_ms = np.percentile(ttfts, 50) * 1000
        p95_ms = np.percentile(ttfts, 95) * 1000

        print(f"{seq_len:>10} {prompt_tokens:>12} {cached_tokens:>12} {extend_tokens:>12} "
              f"{mean_ms:>14.1f} {p50_ms:>13.1f} {p95_ms:>13.1f}")

        results.append({
            "seq_len": seq_len,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "extend_tokens": extend_tokens,
            "ttft_mean": np.mean(ttfts),
            "ttft_p50": np.percentile(ttfts, 50),
            "ttft_p95": np.percentile(ttfts, 95),
            "ttfts": ttfts,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Quick linear fit check
    if len(results) >= 3:
        et = np.array([r["extend_tokens"] for r in results])
        ttft = np.array([r["ttft_p50"] for r in results])
        # Linear fit
        X1 = np.column_stack([et, np.ones(len(results))])
        t1, *_ = np.linalg.lstsq(X1, ttft, rcond=None)
        pred1 = X1 @ t1
        r2_lin = 1 - np.sum((ttft - pred1)**2) / np.sum((ttft - np.mean(ttft))**2)
        # Quadratic fit
        X2 = np.column_stack([et**2, et, np.ones(len(results))])
        t2, *_ = np.linalg.lstsq(X2, ttft, rcond=None)
        pred2 = X2 @ t2
        r2_quad = 1 - np.sum((ttft - pred2)**2) / np.sum((ttft - np.mean(ttft))**2)
        print(f"\n=== Pure Prefill Fit (P50) ===")
        print(f"  Linear  R² = {r2_lin:.4f}  (α={t1[0]*1000:.4f} ms/token)")
        print(f"  Quadratic R² = {r2_quad:.4f}  (η={t2[0]*1e6:.6f} ms/token²)")


if __name__ == "__main__":
    main()
