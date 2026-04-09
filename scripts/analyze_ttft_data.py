"""
Offline analysis script for TTFT prediction data collection.

Reads the JSONL files exported by --export-metrics-to-file and analyzes
the relationship between TTFT, extend_tokens, cached_tokens, and queue_time.

Usage:
    python scripts/analyze_ttft_data.py --log-dir /tmp/sglang_metrics
"""

import argparse
import glob
import json
import os

import numpy as np


def load_records(log_dir: str):
    pattern = os.path.join(log_dir, "sglang-request-metrics-*.log")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No metrics log files found in {log_dir}")
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(records)} records from {len(files)} file(s).")
    return records


def extract_features(records):
    rows = []
    for r in records:
        ttft = r.get("ttft")
        extend_tokens = r.get("extend_tokens")
        if ttft is None or extend_tokens is None:
            continue
        cached_tokens = r.get("cached_tokens", 0) or 0
        prompt_tokens = r.get("prompt_tokens", 0) or 0
        rows.append(
            {
                "ttft": ttft,
                "extend_tokens": extend_tokens,
                "cached_tokens": cached_tokens,
                "prompt_tokens": prompt_tokens,
                "cache_hit_rate": (
                    cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
                ),
                "queue_time": r.get("queue_time"),
                "e2e_latency": r.get("e2e_latency"),
                "inference_time": r.get("inference_time"),
            }
        )
    print(f"Extracted {len(rows)} complete records for analysis.")
    return rows


def print_summary_stats(rows):
    ttft = np.array([r["ttft"] for r in rows])
    extend_tokens = np.array([r["extend_tokens"] for r in rows])
    cached_tokens = np.array([r["cached_tokens"] for r in rows])
    queue_times = np.array(
        [r["queue_time"] if r["queue_time"] is not None else float("nan") for r in rows]
    )
    cache_hit_rates = np.array([r["cache_hit_rate"] for r in rows])

    print("\n=== Summary Statistics ===")
    for name, arr in [
        ("TTFT (s)", ttft),
        ("extend_tokens", extend_tokens),
        ("cached_tokens", cached_tokens),
        ("queue_time (s)", queue_times),
    ]:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            print(f"  {name}: no valid data")
            continue
        print(
            f"  {name:20s}: "
            f"min={np.min(valid):.4f}, "
            f"P50={np.percentile(valid, 50):.4f}, "
            f"P95={np.percentile(valid, 95):.4f}, "
            f"P99={np.percentile(valid, 99):.4f}, "
            f"max={np.max(valid):.4f}, "
            f"mean={np.mean(valid):.4f}"
        )
    print(
        f"  {'cache_hit_rate':20s}: "
        f"mean={np.mean(cache_hit_rates):.3f}, "
        f"P50={np.percentile(cache_hit_rates, 50):.3f}"
    )

    print("\n=== Pearson Correlation with TTFT ===")
    for name, arr in [
        ("extend_tokens", extend_tokens),
        ("cached_tokens", cached_tokens),
        ("queue_time", queue_times),
    ]:
        valid_mask = ~np.isnan(arr) & ~np.isnan(ttft)
        if np.sum(valid_mask) > 1:
            corr = np.corrcoef(ttft[valid_mask], arr[valid_mask])[0, 1]
            print(f"  corr(TTFT, {name:20s}) = {corr:.4f}")


def fit_linear_model(rows):
    """Fit OLS: ttft = alpha * extend_tokens + beta * queue_time + gamma."""
    ttft = np.array([r["ttft"] for r in rows])
    extend_tokens = np.array([r["extend_tokens"] for r in rows])
    queue_time = np.array(
        [r["queue_time"] if r["queue_time"] is not None else 0.0 for r in rows]
    )
    X = np.column_stack([extend_tokens, queue_time, np.ones(len(rows))])
    try:
        theta, _, _, _ = np.linalg.lstsq(X, ttft, rcond=None)
    except np.linalg.LinAlgError as e:
        print(f"Linear regression failed: {e}")
        return

    alpha, beta, gamma = theta
    pred = X @ theta
    ss_res = np.sum((ttft - pred) ** 2)
    ss_tot = np.sum((ttft - np.mean(ttft)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(ttft - pred))
    p50 = np.percentile(np.abs(ttft - pred), 50)
    p95 = np.percentile(np.abs(ttft - pred), 95)

    print("\n=== Linear Model: TTFT = α·extend_tokens + β·queue_time + γ ===")
    print(f"  α (per-token prefill cost) = {alpha * 1000:.4f} ms/token")
    print(f"  β (queue time coefficient) = {beta:.4f}")
    print(f"  γ (base latency)           = {gamma * 1000:.2f} ms")
    print(f"  R²                         = {r2:.4f}")
    print(f"  MAE                        = {mae * 1000:.2f} ms")
    print(f"  |error| P50                = {p50 * 1000:.2f} ms")
    print(f"  |error| P95                = {p95 * 1000:.2f} ms")


def analyze_by_extend_tokens_bucket(rows):
    buckets = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, float("inf")]
    bucket_ttft = {i: [] for i in range(len(buckets) - 1)}

    for r in rows:
        et = r["extend_tokens"]
        for i in range(len(buckets) - 1):
            if buckets[i] <= et < buckets[i + 1]:
                bucket_ttft[i].append(r["ttft"])
                break

    print("\n=== TTFT by extend_tokens Bucket ===")
    for i in range(len(buckets) - 1):
        data = bucket_ttft[i]
        if not data:
            continue
        arr = np.array(data)
        lo, hi = buckets[i], buckets[i + 1]
        label = f"[{int(lo)}, {int(hi) if hi != float('inf') else '∞'})"
        print(
            f"  {label:>20}: N={len(arr):5d}, "
            f"mean={np.mean(arr)*1000:.1f}ms, "
            f"P50={np.percentile(arr, 50)*1000:.1f}ms, "
            f"P95={np.percentile(arr, 95)*1000:.1f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze TTFT prediction data")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/sglang_metrics",
        help="Directory containing sglang-request-metrics-*.log files",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=10,
        help="Minimum number of records required for analysis",
    )
    args = parser.parse_args()

    records = load_records(args.log_dir)
    rows = extract_features(records)

    if len(rows) < args.min_records:
        print(f"Not enough complete records ({len(rows)} < {args.min_records}). Exiting.")
        return

    print_summary_stats(rows)
    fit_linear_model(rows)
    analyze_by_extend_tokens_bucket(rows)


if __name__ == "__main__":
    main()
