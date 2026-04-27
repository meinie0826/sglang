#!/usr/bin/env python3
"""Synthetic stage-1 validation for window-level p50 TTFT prediction.

This script simulates a request stream, aggregates per-window features, and
evaluates the formulation:

    features(window x) -> p50_ttft(window x)

Usage:
    python3 scripts/test_window_ttft_synthetic.py
    python3 scripts/test_window_ttft_synthetic.py --window-seconds 120 --duration-seconds 14400
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


@dataclass
class RequestRecord:
    arrival_ts: float
    finish_ts: float
    seqlen: int
    cachelen: int
    extend_tokens: int
    ttft_ms: float


def make_qps_profile(total_windows: int, rng: np.random.Generator) -> np.ndarray:
    base = np.zeros(total_windows, dtype=np.float64)
    qps = rng.uniform(1.5, 3.5)
    for idx in range(total_windows):
        # Piecewise drift + periodicity + occasional bursts.
        if idx % 24 == 0:
            qps += rng.normal(0.0, 0.35)
        qps += rng.normal(0.0, 0.05)
        periodic = 0.8 * math.sin(idx / 8.0) + 0.35 * math.sin(idx / 2.7)
        burst = rng.uniform(1.0, 2.2) if rng.random() < 0.08 else 0.0
        base[idx] = max(0.4, qps + periodic + burst)

    # Introduce workload drift in the second half.
    drift_start = total_windows // 2
    base[drift_start:] *= 1.15
    return base


def simulate_requests(
    qps_profile: Sequence[float],
    window_seconds: float,
    rng: np.random.Generator,
) -> List[RequestRecord]:
    requests: List[RequestRecord] = []
    prev_window_qps = float(qps_profile[0]) if len(qps_profile) > 0 else 1.0

    for window_idx, qps in enumerate(qps_profile):
        window_start = window_idx * window_seconds
        req_count = rng.poisson(max(qps, 0.01) * window_seconds)
        if req_count <= 0:
            prev_window_qps = float(qps)
            continue

        arrival_offsets = np.sort(rng.uniform(0.0, window_seconds, size=req_count))
        prior_active = 0
        for offset in arrival_offsets:
            arrival_ts = window_start + float(offset)
            regime = rng.choice([0, 1, 2], p=[0.62, 0.28, 0.10])
            if regime == 0:
                seqlen = int(np.clip(rng.lognormal(6.2, 0.45), 256, 8192))
                cache_ratio = rng.uniform(0.08, 0.55)
            elif regime == 1:
                seqlen = int(np.clip(rng.lognormal(7.3, 0.38), 2048, 32768))
                cache_ratio = rng.uniform(0.20, 0.78)
            else:
                seqlen = int(np.clip(rng.lognormal(8.05, 0.28), 8192, 65536))
                cache_ratio = rng.uniform(0.35, 0.90)

            cachelen = int(seqlen * cache_ratio)
            extend_tokens = max(1, seqlen - cachelen)

            # Service curve: prompt cost + cache-hit relief + queuing + nonlinear load.
            service_ms = (
                35.0
                + 0.028 * extend_tokens
                + 0.0015 * seqlen
                - 0.0010 * cachelen
            )
            queue_ms = 16.0 * prior_active + 9.0 * max(qps - 3.0, 0.0)
            qps_shift_ms = 5.5 * max(qps - prev_window_qps, 0.0)

            # Simulate workload drift after halfway point.
            if window_idx >= len(qps_profile) // 2:
                service_ms *= 1.12
                queue_ms *= 1.10

            noise_ms = rng.normal(0.0, 18.0 + 0.004 * extend_tokens)
            ttft_ms = max(8.0, service_ms + queue_ms + qps_shift_ms + noise_ms)
            finish_ts = arrival_ts + ttft_ms / 1000.0
            requests.append(
                RequestRecord(
                    arrival_ts=arrival_ts,
                    finish_ts=finish_ts,
                    seqlen=seqlen,
                    cachelen=cachelen,
                    extend_tokens=extend_tokens,
                    ttft_ms=ttft_ms,
                )
            )
            prior_active += 1
        prev_window_qps = float(qps)

    return requests


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def build_window_snapshots(
    requests: Sequence[RequestRecord],
    qps_profile: Sequence[float],
    window_seconds: float,
    min_requests: int,
) -> List[dict]:
    snapshots: List[dict] = []
    requests_by_finish = sorted(requests, key=lambda r: r.finish_ts)

    for window_idx, current_qps in enumerate(qps_profile):
        window_end = (window_idx + 1) * window_seconds
        window_start = max(0.0, window_end - window_seconds)
        finished = [
            r
            for r in requests_by_finish
            if window_start <= r.finish_ts < window_end
        ]
        if len(finished) < min_requests:
            continue

        seqlens = np.array([r.seqlen for r in finished], dtype=np.float64)
        cachelens = np.array([r.cachelen for r in finished], dtype=np.float64)
        extend_tokens = np.array([r.extend_tokens for r in finished], dtype=np.float64)
        ttfts = np.array([r.ttft_ms for r in finished], dtype=np.float64)
        arrivals = sum(
            1
            for r in requests
            if window_start <= r.arrival_ts < window_end
        )
        active_at_end = sum(
            1 for r in requests if r.arrival_ts < window_end <= r.finish_ts
        )
        snapshots.append(
            {
                "window_idx": window_idx,
                "window_start": window_start,
                "window_end": window_end,
                "features": {
                    "active_requests": float(active_at_end),
                    "arrival_qps_60s": float(arrivals / window_seconds),
                    "finished_qps_60s": float(len(finished) / window_seconds),
                    "finished_count_60s": float(len(finished)),
                    "window_mean_seqlen_60s": float(seqlens.mean()),
                    "window_p90_seqlen_60s": percentile(seqlens, 90),
                    "window_mean_cachelen_60s": float(cachelens.mean()),
                    "window_p90_cachelen_60s": percentile(cachelens, 90),
                    "window_mean_extend_tokens_60s": float(extend_tokens.mean()),
                    "window_p90_extend_tokens_60s": percentile(extend_tokens, 90),
                    "window_mean_ttft_ms_60s": float(ttfts.mean()),
                    "window_p90_ttft_ms_60s": percentile(ttfts, 90),
                },
                "window_p50_ttft_ms": percentile(ttfts, 50),
                "current_qps": float(current_qps),
            }
        )

    return snapshots


FEATURE_NAMES = [
    "active_requests",
    "arrival_qps_60s",
    "finished_qps_60s",
    "finished_count_60s",
    "window_mean_seqlen_60s",
    "window_p90_seqlen_60s",
    "window_mean_cachelen_60s",
    "window_p90_cachelen_60s",
    "window_mean_extend_tokens_60s",
    "window_p90_extend_tokens_60s",
    "window_mean_ttft_ms_60s",
    "window_p90_ttft_ms_60s",
]


def build_supervised_rows(snapshots: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for snapshot in snapshots:
        row_features = dict(snapshot["features"])
        rows.append(
            {
                "window_idx": int(snapshot["window_idx"]),
                "features": row_features,
                "label": float(snapshot["window_p50_ttft_ms"]),
            }
        )
    return rows


def vectorize(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    y = np.array([float(row["label"]) for row in rows], dtype=np.float64)
    return x, y


def fit_xgboost(x_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is required for the synthetic TTFT validation script.")
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=96,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=1,
    )
    model.fit(x_train.astype(np.float32), y_train.astype(np.float32), verbose=False)
    return model


def predict(x: np.ndarray, model: XGBRegressor) -> np.ndarray:
    return model.predict(x.astype(np.float32))


def summarize_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    mape = np.mean(abs_err / np.maximum(y_true, 1e-6)) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return {
        "mae_ms": float(np.mean(abs_err)),
        "p50_abs_err_ms": percentile(abs_err, 50),
        "p90_abs_err_ms": percentile(abs_err, 90),
        "mape_pct": float(mape),
        "r2": float(r2),
    }


def time_split(rows: Sequence[dict], train_ratio: float, valid_ratio: float) -> tuple[list, list, list]:
    n = len(rows)
    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, n)
    return list(rows[:train_end]), list(rows[train_end:valid_end]), list(rows[valid_end:])


def dump_dataset(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=int, default=6 * 3600)
    parser.add_argument("--min-window-requests", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--dump-jsonl", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    total_windows = int(args.duration_seconds / args.window_seconds)
    qps_profile = make_qps_profile(total_windows, rng)
    requests = simulate_requests(qps_profile, args.window_seconds, rng)
    snapshots = build_window_snapshots(
        requests,
        qps_profile,
        args.window_seconds,
        args.min_window_requests,
    )
    rows = build_supervised_rows(snapshots)
    if len(rows) < 30:
        raise SystemExit("Not enough synthetic rows generated; increase duration.")

    if args.dump_jsonl:
        dump_dataset(Path(args.dump_jsonl), rows)

    train_rows, valid_rows, test_rows = time_split(
        rows, args.train_ratio, args.valid_ratio
    )
    x_train, y_train = vectorize(train_rows)
    x_valid, y_valid = vectorize(valid_rows)
    x_test, y_test = vectorize(test_rows)
    model = fit_xgboost(x_train, y_train)
    valid_pred = predict(x_valid, model)
    test_pred = predict(x_test, model)

    print("Synthetic window-TTFT validation")
    print(f"  windows_total: {total_windows}")
    print(f"  requests_total: {len(requests)}")
    print(f"  supervised_rows: {len(rows)}")
    print(f"  split: train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)}")
    print("")

    valid_stats = summarize_errors(y_valid, valid_pred)
    test_stats = summarize_errors(y_test, test_pred)
    print("Validation")
    for key, value in valid_stats.items():
        print(f"  {key}: {value:.3f}")
    print("")
    print("Test")
    for key, value in test_stats.items():
        print(f"  {key}: {value:.3f}")

    latest = rows[-1]
    base = np.array(
        [[float(latest["features"][name]) for name in FEATURE_NAMES]], dtype=np.float64
    )
    print("")
    print("What-if on latest window state")
    qps_idx = FEATURE_NAMES.index("arrival_qps_60s")
    for future_qps in [2.0, 4.0, 6.0, 8.0, 12.0]:
        scenario = base.copy()
        scenario[0, qps_idx] = future_qps
        scenario[0, FEATURE_NAMES.index("finished_qps_60s")] = future_qps
        pred = float(predict(scenario, model)[0])
        print(f"  qps={future_qps:>4.1f} -> predicted_window_p50_ttft_ms={pred:>8.2f}")


if __name__ == "__main__":
    main()
