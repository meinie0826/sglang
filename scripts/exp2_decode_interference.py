"""
Experiment 2: Decode Interference
====================================
Fix prompt length, vary number of concurrent decode requests.
Measures how decode concurrency affects TTFT.

Strategy:
  1. Send N long-output background requests (max_new_tokens=512) that stay in decode phase.
  2. Wait until all background requests are in decode phase (first token returned).
  3. Send a probe request with fixed prompt_len, measure TTFT.
  4. Repeat for n_decode in [0, 2, 4, 8, 16, 32].

Usage:
    python scripts/exp2_decode_interference.py --host 127.0.0.1 --port 30000
"""

import argparse
import json
import threading
import time

import numpy as np
import requests

PROBE_SEQ_LEN = 4096           # Fixed prompt length for probe
BACKGROUND_OUTPUT_LEN = 512    # Long enough to stay in decode during probe
N_DECODE_LEVELS = [0, 2, 4, 8, 16, 32]
PROBE_REPEATS = 3


def send_request(host: str, port: int, prompt_len: int, max_new_tokens: int,
                 stream: bool = False, timeout: int = 600):
    import random
    input_ids = [random.randint(100, 50000) for _ in range(prompt_len)]
    url = f"http://{host}:{port}/generate"
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        "stream": stream,
        "log_metrics": True,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def send_streaming_and_wait_first_token(host: str, port: int, prompt_len: int,
                                         max_new_tokens: int, first_token_event: threading.Event):
    """Send a streaming request, set event when first token arrives, then continue."""
    import random
    input_ids = [random.randint(100, 50000) for _ in range(prompt_len)]
    url = f"http://{host}:{port}/generate"
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        "stream": True,
        "log_metrics": False,
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as resp:
            for chunk in resp.iter_lines():
                if chunk:
                    first_token_event.set()  # First token received, now in decode phase
                    # Keep consuming to stay alive (don't break early)
    except Exception:
        first_token_event.set()  # Ensure event is set even on error


def run_with_n_decode(host: str, port: int, n_decode: int, probe_seq_len: int) -> list:
    """Run probe requests with n_decode background decode requests."""
    ttfts = []

    if n_decode == 0:
        # No background load
        for _ in range(PROBE_REPEATS):
            t0 = time.perf_counter()
            data = send_request(host, port, probe_seq_len, max_new_tokens=1)
            t1 = time.perf_counter()
            meta = data.get("meta_info", {})
            ttft = meta.get("ttft") or (t1 - t0)
            ttfts.append(ttft)
            time.sleep(1.0)
    else:
        for _ in range(PROBE_REPEATS):
            # Launch N background decode requests
            events = [threading.Event() for _ in range(n_decode)]
            threads = []
            for evt in events:
                t = threading.Thread(
                    target=send_streaming_and_wait_first_token,
                    args=(host, port, 2048, BACKGROUND_OUTPUT_LEN, evt),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            # Wait until all background requests have received first token (in decode phase)
            for evt in events:
                evt.wait(timeout=60)

            # Small extra wait to ensure all are firmly in decode
            time.sleep(0.5)

            # Send probe request
            t0 = time.perf_counter()
            data = send_request(host, port, probe_seq_len, max_new_tokens=1)
            t1 = time.perf_counter()
            meta = data.get("meta_info", {})
            ttft = meta.get("ttft") or (t1 - t0)
            ttfts.append(ttft)

            # Wait for background to finish
            for t in threads:
                t.join(timeout=120)

            time.sleep(2.0)  # Cooldown

    return ttfts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--probe-seq-len", type=int, default=PROBE_SEQ_LEN)
    parser.add_argument("--output", default="exp2_decode_interference.json")
    args = parser.parse_args()

    probe_seq_len = args.probe_seq_len

    results = []
    print(f"Probe seq_len={probe_seq_len}, background_output_len={BACKGROUND_OUTPUT_LEN}")
    print(f"{'n_decode':>10} {'ttft_mean(ms)':>14} {'ttft_p50(ms)':>13} {'ttft_p95(ms)':>13}")
    print("-" * 55)

    for n_decode in N_DECODE_LEVELS:
        print(f"  Running n_decode={n_decode}...", end="", flush=True)
        ttfts = run_with_n_decode(args.host, args.port, n_decode, probe_seq_len)
        mean_ms = np.mean(ttfts) * 1000
        p50_ms = np.percentile(ttfts, 50) * 1000
        p95_ms = np.percentile(ttfts, 95) * 1000
        print(f"\r{n_decode:>10} {mean_ms:>14.1f} {p50_ms:>13.1f} {p95_ms:>13.1f}")
        results.append({
            "n_decode": n_decode,
            "probe_seq_len": probe_seq_len,
            "ttft_mean": np.mean(ttfts),
            "ttft_p50": np.percentile(ttfts, 50),
            "ttft_p95": np.percentile(ttfts, 95),
            "ttfts": ttfts,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Fit interference model
    if len(results) >= 3:
        nd = np.array([r["n_decode"] for r in results])
        ttft = np.array([r["ttft_p50"] for r in results])
        base = ttft[nd == 0][0] if (nd == 0).any() else ttft[0]
        print(f"\n=== Decode Interference (P50, probe_seq_len={probe_seq_len}) ===")
        print(f"  Base TTFT (n_decode=0): {base*1000:.1f} ms")
        for r in results:
            overhead = (r["ttft_p50"] - base) * 1000
            print(f"  n_decode={r['n_decode']:>3}: TTFT={r['ttft_p50']*1000:.1f}ms, overhead={overhead:+.1f}ms")

        # Linear fit of overhead vs n_decode
        overhead = ttft - base
        X = np.column_stack([nd, np.ones(len(nd))])
        t, *_ = np.linalg.lstsq(X, overhead, rcond=None)
        pred = X @ t
        r2 = 1 - np.sum((overhead - pred)**2) / max(np.sum((overhead - np.mean(overhead))**2), 1e-12)
        print(f"\n  Linear fit: overhead = {t[0]*1000:.1f}ms * n_decode + {t[1]*1000:.1f}ms, R²={r2:.4f}")


if __name__ == "__main__":
    main()
