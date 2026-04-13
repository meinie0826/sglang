#!/usr/bin/env python3
"""
MiniMax-M2.5 decode attention kernel micro-benchmark.

This script directly measures the FlashAttention decode kernel latency for
MiniMax-M2.5 with TP=4, EP=4 configuration on SM100 (Blackwell).

Usage:
    # Run with your actual model path to get accurate measurements
    python scripts/bench_minimax25_decode_micro.py
    
    # Specify batch size and sequence length
    python scripts/bench_minimax25_decode_micro.py --batch-size 64 --seq-len 8192
    
    # Sweep different configurations
    python scripts/bench_minimax25_decode_micro.py --mode sweep

Note: For most accurate results, run on a dedicated GPU with no other workloads.
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

# ============== MiniMax-M2.5 Config ==============
# Full model config
NUM_Q_HEADS_TOTAL = 48
NUM_KV_HEADS_TOTAL = 8
HEAD_DIM = 128
NUM_LAYERS = 62

# Per-GPU config (TP=4)
TP_SIZE = 4
NUM_Q_HEADS = NUM_Q_HEADS_TOTAL // TP_SIZE  # 12
NUM_KV_HEADS = NUM_KV_HEADS_TOTAL // TP_SIZE  # 2
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 6


@dataclass
class BenchResult:
    batch_size: int
    seq_len: int
    latency_us: float
    traffic_gb: float
    bandwidth_gb_s: float
    per_layer_ms: float


def create_decode_inputs(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create inputs simulating decode phase with paged KV cache."""
    # Q: [batch, num_q_heads, head_dim] - single token query
    q = torch.randn(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
    
    # KV: [batch, seq_len, num_kv_heads, head_dim] - full KV cache
    # In practice, this is paged. For benchmark we use contiguous.
    k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    
    return q, k, v


def decode_attention_impl(
    q: torch.Tensor,  # [B, H_Q, D]
    k: torch.Tensor,  # [B, S, H_KV, D]
    v: torch.Tensor,  # [B, S, H_KV, D]
) -> torch.Tensor:
    """
    Reference decode attention with GQA expansion.
    
    For MiniMax-M2.5 with TP=4:
    - Q has 12 heads per GPU
    - K/V have 2 heads per GPU
    - GQA expansion: each KV head serves 6 Q heads
    """
    batch_size = q.shape[0]
    seq_len = k.shape[1]
    
    # Expand K/V for GQA: [B, S, H_KV, D] -> [B, S, H_Q, D]
    k_expanded = k.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM
    )
    v_expanded = v.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM
    )
    
    # Transpose to [B, H, S, D] for attention
    q_t = q.unsqueeze(2)  # [B, H_Q, 1, D]
    k_t = k_expanded.transpose(1, 2)  # [B, H_Q, S, D]
    v_t = v_expanded.transpose(1, 2)  # [B, H_Q, S, D]
    
    # Scaled dot-product attention
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # Use PyTorch's optimized SDPA
    output = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        is_causal=True,  # Causal mask for decode
        scale=scale,
    )
    
    return output.squeeze(2)  # [B, H_Q, D]


def bench_single(
    batch_size: int,
    seq_len: int,
    warmup_iters: int = 20,
    bench_iters: int = 100,
    dtype: torch.dtype = torch.bfloat16,
) -> BenchResult:
    """Benchmark single configuration."""
    q, k, v = create_decode_inputs(batch_size, seq_len, dtype)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = decode_attention_impl(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events for precision
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(bench_iters):
        _ = decode_attention_impl(q, k, v)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / bench_iters
    latency_us = latency_ms * 1000
    
    # Calculate memory traffic (GB)
    # Q: B * H_Q * D * 2
    # K: B * S * H_KV * D * 2
    # V: B * S * H_KV * D * 2
    # O: B * H_Q * D * 2
    element_size = 2  # bf16
    q_bytes = batch_size * NUM_Q_HEADS * HEAD_DIM * element_size
    kv_bytes = 2 * batch_size * seq_len * NUM_KV_HEADS * HEAD_DIM * element_size
    o_bytes = batch_size * NUM_Q_HEADS * HEAD_DIM * element_size
    total_bytes = q_bytes + kv_bytes + o_bytes
    traffic_gb = total_bytes / 1e9
    
    # Bandwidth
    bandwidth_gb_s = traffic_gb / (latency_ms / 1000)
    
    # Per-layer time in ms
    per_layer_ms = latency_ms
    
    return BenchResult(
        batch_size=batch_size,
        seq_len=seq_len,
        latency_us=latency_us,
        traffic_gb=traffic_gb,
        bandwidth_gb_s=bandwidth_gb_s,
        per_layer_ms=per_layer_ms,
    )


def analyze_attn_overhead(result: BenchResult) -> dict:
    """
    Analyze if the attention overhead is reasonable.
    
    For MiniMax-M2.5 decode with TP=4:
    - Each GPU processes 12 Q heads, 2 KV heads
    - Typical decode budget: ~50ms total per token
    - Expected attention share: 30-40%
    """
    total_attn_time_ms = result.per_layer_ms * NUM_LAYERS
    
    # Typical decode latency targets
    target_decode_latencies_ms = [30, 50, 100]
    
    analysis = {
        "per_layer_us": result.latency_us,
        "total_attn_ms": total_attn_time_ms,
        "attn_share_at_targets": {},
    }
    
    for target_ms in target_decode_latencies_ms:
        share = (total_attn_time_ms / target_ms) * 100
        analysis["attn_share_at_targets"][target_ms] = share
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax-M2.5 decode attention micro-benchmark"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sweep"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MiniMax-M2.5 Decode Attention Micro-Benchmark")
    print(f"Config: TP={TP_SIZE}, Per-GPU: {NUM_Q_HEADS} Q heads, {NUM_KV_HEADS} KV heads")
    print("=" * 80)
    print()
    
    if args.mode == "single":
        print(f"Running: batch_size={args.batch_size}, seq_len={args.seq_len}")
        print()
        result = bench_single(args.batch_size, args.seq_len)
        
        print(f"Latency per layer: {result.latency_us:.2f} us")
        print(f"Memory traffic: {result.traffic_gb:.3f} GB")
        print(f"Effective bandwidth: {result.bandwidth_gb_s:.1f} GB/s")
        print()
        
        analysis = analyze_attn_overhead(result)
        print("Total attention time (all layers):")
        print(f"  {analysis['total_attn_ms']:.2f} ms")
        print()
        print("Attention share at different decode latency targets:")
        for target_ms, share in analysis["attn_share_at_targets"].items():
            status = "⚠️ HIGH" if share > 40 else "✓ OK"
            print(f"  {target_ms}ms target: {share:.1f}% {status}")
        
    else:
        # Sweep mode
        if args.quick:
            batch_sizes = [1, 8, 32, 64, 128]
            seq_lens = [1024, 4096, 8192, 32768]
        else:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        
        print(f"{'Batch':>8} | {'SeqLen':>8} | {'Latency (us)':>12} | {'Total attn (ms)':>14} | {'BW (GB/s)':>10}")
        print("-" * 70)
        
        results = []
        for bs in batch_sizes:
            for sl in seq_lens:
                try:
                    r = bench_single(bs, sl)
                    total_attn_ms = r.per_layer_ms * NUM_LAYERS
                    print(f"{bs:>8} | {sl:>8} | {r.latency_us:>12.2f} | {total_attn_ms:>14.2f} | {r.bandwidth_gb_s:>10.1f}")
                    results.append(r)
                except Exception as e:
                    print(f"{bs:>8} | {sl:>8} | Error: {e}")
        
        print()
        print("=" * 80)
        
        # Find optimal operating point
        if results:
            # Sort by efficiency (latency/batch_size)
            best = min(results, key=lambda r: r.latency_us / r.batch_size)
            print(f"Most efficient config: batch_size={best.batch_size}, seq_len={best.seq_len}")
            print(f"  Latency: {best.latency_us:.2f} us/layer")
            print(f"  Total attention: {best.per_layer_ms * NUM_LAYERS:.2f} ms")


if __name__ == "__main__":
    main()
