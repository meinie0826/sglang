#!/usr/bin/env python3
"""
Direct kernel benchmark for MiniMax-M2.5 decode attention on SM100 (Blackwell).

This script directly benchmarks the FlashAttention kernel used in MiniMax-M2.5
decode phase. It uses CUDA events for precise timing.

MiniMax-M2.5 model config:
- num_q_heads = 48
- num_kv_heads = 8 (GQA ratio 6:1)
- head_dim = 128
- num_layers = 62

Target kernel: fmhaSm100fKernel QkvE4m30Bfloat16H128PagedKvCausalP64VarSegQ8Kv128PersistentSwapsAbForGen

Usage:
    python scripts/bench_minimax25_attn_kernel.py
    
    # With specific batch size and sequence length
    python scripts/bench_minimax25_attn_kernel.py --batch-size 64 --seq-len 8192
    
    # Sweep mode
    python scripts/bench_minimax25_attn_kernel.py --mode sweep
    
    # Quick test
    python scripts/bench_minimax25_attn_kernel.py --quick
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# MiniMax-M2.5 model config (full model)
NUM_Q_HEADS_TOTAL = 48
NUM_KV_HEADS_TOTAL = 8
HEAD_DIM = 128
NUM_LAYERS = 62

# Per-GPU config (TP=4, EP=4)
TP_SIZE = 4
EP_SIZE = 4
NUM_Q_HEADS = NUM_Q_HEADS_TOTAL // TP_SIZE  # 12 per GPU
NUM_KV_HEADS = NUM_KV_HEADS_TOTAL // TP_SIZE  # 2 per GPU

# GQA expansion ratio (remains the same)
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 6


@dataclass
class BenchmarkResult:
    batch_size: int
    seq_len: int
    latency_us: float
    throughput_gb_s: float
    theoretical_min_us: float
    efficiency_ratio: float


def check_sm100():
    """Check if running on SM100 (Blackwell) architecture."""
    major, minor = torch.cuda.get_device_capability()
    return major == 10  # SM100 is compute capability 10.x


def create_paged_kv_cache(
    batch_size: int,
    seq_len: int,
    page_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create paged KV cache for decode attention."""
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    
    # KV cache: [total_pages, page_size, num_kv_heads, head_dim]
    k_cache = torch.randn(total_pages, page_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v_cache = torch.randn(total_pages, page_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    
    # Block table: [batch_size, num_pages_per_seq]
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages_per_seq
    )
    
    return k_cache, v_cache, block_table


def create_decode_query(
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> torch.Tensor:
    """Create query tensor for decode (single token per sequence)."""
    # Query: [batch_size, 1, num_q_heads, head_dim]
    return torch.randn(batch_size, 1, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)


def expand_kv_for_gqa(kv_cache: torch.Tensor) -> torch.Tensor:
    """Expand KV cache from [batch, seq, num_kv_heads, head_dim] to [batch, seq, num_q_heads, head_dim] for GQA."""
    # Expand along the head dimension
    # [batch, seq, num_kv_heads, head_dim] -> [batch, seq, num_q_heads, head_dim]
    batch, seq_len, _, head_dim = kv_cache.shape
    expanded = kv_cache.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1)
    return expanded.reshape(batch, seq_len, NUM_Q_HEADS, head_dim)


def decode_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    page_size: int,
) -> torch.Tensor:
    """
    Reference implementation of decode attention using PyTorch SDPA.
    
    Args:
        q: [batch_size, 1, num_q_heads, head_dim]
        k_cache: [total_pages, page_size, num_kv_heads, head_dim]
        v_cache: [total_pages, page_size, num_kv_heads, head_dim]
        block_table: [batch_size, num_pages_per_seq]
        seq_len: Current sequence length (number of KV tokens)
        page_size: Size of each page
    
    Returns:
        output: [batch_size, 1, num_q_heads, head_dim]
    """
    batch_size = q.shape[0]
    num_pages_per_seq = block_table.shape[1]
    
    # Gather KV from paged cache
    # [batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim]
    k_paged = k_cache[block_table.flatten()].reshape(
        batch_size, num_pages_per_seq, page_size, NUM_KV_HEADS, HEAD_DIM
    )
    v_paged = v_cache[block_table.flatten()].reshape(
        batch_size, num_pages_per_seq, page_size, NUM_KV_HEADS, HEAD_DIM
    )
    
    # Flatten to [batch_size, seq_len, num_kv_heads, head_dim]
    k_flat = k_paged[:, :seq_len // page_size + 1].reshape(
        batch_size, -1, NUM_KV_HEADS, HEAD_DIM
    )[:, :seq_len]
    v_flat = v_paged[:, :seq_len // page_size + 1].reshape(
        batch_size, -1, NUM_KV_HEADS, HEAD_DIM
    )[:, :seq_len]
    
    # Expand KV for GQA: [batch_size, seq_len, num_q_heads, head_dim]
    k_expanded = expand_kv_for_gqa(k_flat)
    v_expanded = expand_kv_for_gqa(v_flat)
    
    # Transpose for SDPA: [batch_size, num_q_heads, 1, head_dim], [batch_size, num_q_heads, seq_len, head_dim]
    q_t = q.transpose(1, 2)  # [batch_size, num_q_heads, 1, head_dim]
    k_t = k_expanded.transpose(1, 2)  # [batch_size, num_q_heads, seq_len, head_dim]
    v_t = v_expanded.transpose(1, 2)  # [batch_size, num_q_heads, seq_len, head_dim]
    
    # Compute scaled dot-product attention with causal mask
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # Create causal mask: [1, 1, 1, seq_len]
    causal_mask = torch.triu(
        torch.ones(1, seq_len, dtype=torch.bool, device=q.device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    output = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=causal_mask,
        scale=scale,
    )
    
    return output.transpose(1, 2)  # [batch_size, 1, num_q_heads, head_dim]


def benchmark_decode_attention(
    batch_size: int,
    seq_len: int,
    page_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 20,
    bench_iters: int = 100,
) -> float:
    """
    Benchmark decode attention kernel using CUDA events.
    
    Returns:
        latency in microseconds
    """
    device = "cuda"
    
    # Create inputs
    q = create_decode_query(batch_size, dtype, device)
    k_cache, v_cache, block_table = create_paged_kv_cache(
        batch_size, seq_len, page_size, dtype, device
    )
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = decode_attention_reference(q, k_cache, v_cache, block_table, seq_len, page_size)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event.record()
    for _ in range(bench_iters):
        _ = decode_attention_reference(q, k_cache, v_cache, block_table, seq_len, page_size)
    end_event.record()
    torch.cuda.synchronize()
    
    latency_ms = start_event.elapsed_time(end_event) / bench_iters
    latency_us = latency_ms * 1000
    
    return latency_us


def estimate_theoretical_latency(batch_size: int, seq_len: int, dtype: torch.dtype = torch.bfloat16):
    """
    Estimate theoretical latency based on memory bandwidth.
    
    For decode attention with GQA:
    - Q read: batch_size * num_q_heads * head_dim * 2 bytes
    - K read: batch_size * seq_len * num_kv_heads * head_dim * 2 bytes
    - V read: batch_size * seq_len * num_kv_heads * head_dim * 2 bytes
    - O write: batch_size * num_q_heads * head_dim * 2 bytes
    
    Memory traffic = Q + K + V + O
                  = batch_size * 2 * (num_q_heads * head_dim + 2 * seq_len * num_kv_heads * head_dim)
    """
    element_size = 2  # bfloat16
    
    # Memory traffic (bytes)
    q_traffic = batch_size * NUM_Q_HEADS * HEAD_DIM * element_size
    kv_traffic = 2 * batch_size * seq_len * NUM_KV_HEADS * HEAD_DIM * element_size
    o_traffic = batch_size * NUM_Q_HEADS * HEAD_DIM * element_size
    
    total_traffic = q_traffic + kv_traffic + o_traffic
    
    # B200 peak memory bandwidth: ~8 TB/s (HBM3e)
    # Conservative estimate: 5 TB/s achievable
    bw_tb_s = 5.0  # TB/s
    
    theoretical_latency_us = total_traffic / (bw_tb_s * 1e12) * 1e6
    
    return {
        "total_traffic_bytes": total_traffic,
        "theoretical_latency_us": theoretical_latency_us,
    }


def run_single_benchmark(
    batch_size: int,
    seq_len: int,
    page_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> BenchmarkResult:
    """Run single benchmark and return result."""
    latency_us = benchmark_decode_attention(
        batch_size=batch_size,
        seq_len=seq_len,
        page_size=page_size,
        dtype=dtype,
    )
    
    theoretical = estimate_theoretical_latency(batch_size, seq_len, dtype)
    
    total_traffic = theoretical["total_traffic_bytes"]
    throughput_gb_s = total_traffic / (latency_us * 1e-6) / 1e9
    
    return BenchmarkResult(
        batch_size=batch_size,
        seq_len=seq_len,
        latency_us=latency_us,
        throughput_gb_s=throughput_gb_s,
        theoretical_min_us=theoretical["theoretical_latency_us"],
        efficiency_ratio=latency_us / theoretical["theoretical_latency_us"],
    )


def run_sweep_benchmark(
    batch_sizes: List[int],
    seq_lens: List[int],
    page_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
):
    """Run benchmark sweep across batch sizes and sequence lengths."""
    results = []
    
    print("=" * 110)
    print(f"MiniMax-M2.5 Decode Attention Kernel Benchmark (SM100/Blackwell)")
    print(f"Config: num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print("=" * 110)
    print()
    
    header = (
        f"{'Batch':>8} | {'SeqLen':>8} | {'Latency (us)':>14} | {'Throughput (GB/s)':>17} | "
        f"{'Theoretical (us)':>17} | {'Efficiency':>11}"
    )
    print(header)
    print("-" * len(header))
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            try:
                result = run_single_benchmark(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    page_size=page_size,
                    dtype=dtype,
                )
                
                print(
                    f"{result.batch_size:>8} | {result.seq_len:>8} | {result.latency_us:>14.2f} | "
                    f"{result.throughput_gb_s:>17.2f} | {result.theoretical_min_us:>17.2f} | "
                    f"{result.efficiency_ratio:>10.2f}x"
                )
                
                results.append(result)
            except Exception as e:
                print(f"{batch_size:>8} | {seq_len:>8} | Error: {e}")
    
    print()
    
    # Summary
    print("=" * 110)
    print("Summary:")
    print(f"- Tested {len(results)} configurations")
    print(f"- Configuration: TP={TP_SIZE}, EP={EP_SIZE}")
    print(f"- Per-GPU: {NUM_Q_HEADS} Q heads, {NUM_KV_HEADS} KV heads")
    
    if results:
        avg_efficiency = sum(r.efficiency_ratio for r in results) / len(results)
        max_latency = max(r.latency_us for r in results)
        min_latency = min(r.latency_us for r in results)
        
        print(f"- Average efficiency ratio: {avg_efficiency:.2f}x")
        print(f"- Latency range: {min_latency:.2f} - {max_latency:.2f} us")
    
    print()
    
    # Total attention time estimation
    print("Per-Token Attention Time Estimation:")
    print("-" * 40)
    
    # Find a typical online serving config
    typical_batch = 64
    typical_seq = 8192
    
    typical_result = next(
        (r for r in results if r.batch_size == typical_batch and r.seq_len == typical_seq),
        None,
    )
    
    if typical_result:
        per_layer_time_us = typical_result.latency_us
        total_attn_time_ms = per_layer_time_us * NUM_LAYERS / 1000
        
        print(f"Typical config: batch_size={typical_batch}, seq_len={typical_seq}")
        print(f"Per-layer attention time: {per_layer_time_us:.2f} us")
        print(f"Total attention time ({NUM_LAYERS} layers): {total_attn_time_ms:.2f} ms")
        print()
        
        print("Budget analysis:")
        print("- If target decode latency is 50ms/token")
        print("- And attention should be ~30-40% of total time")
        print(f"- Budget is ~15-20ms")
        
        if total_attn_time_ms > 25:
            print(f"⚠️  ATTENTION TIME ({total_attn_time_ms:.2f} ms) EXCEEDS BUDGET!")
            print("   This confirms the 40% attention overhead you observed.")
        elif total_attn_time_ms > 15:
            print(f"⚠️  Attention time ({total_attn_time_ms:.2f} ms) is at upper limit of budget.")
        else:
            print(f"✓ Attention time ({total_attn_time_ms:.2f} ms) within budget.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MiniMax-M2.5 decode attention kernel"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for single benchmark",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length for single benchmark",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="Page size for paged attention",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "sweep"],
        help="Benchmark mode: single or sweep",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with fewer configs",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=100,
        help="Benchmark iterations",
    )
    args = parser.parse_args()
    
    # Check SM100
    if not check_sm100():
        major, minor = torch.cuda.get_device_capability()
        print(f"Warning: Not running on SM100 (compute capability {major}.{minor})")
        print("Results may not reflect actual kernel performance.")
        print()
    
    # Run benchmark
    if args.mode == "single":
        print(f"Running single benchmark: batch_size={args.batch_size}, seq_len={args.seq_len}")
        print()
        result = run_single_benchmark(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            page_size=args.page_size,
        )
        print(f"Latency: {result.latency_us:.2f} us")
        print(f"Throughput: {result.throughput_gb_s:.2f} GB/s")
        print(f"Theoretical min: {result.theoretical_min_us:.2f} us")
        print(f"Efficiency ratio: {result.efficiency_ratio:.2f}x")
    else:
        if args.quick:
            batch_sizes = [1, 8, 64, 128]
            seq_lens = [1024, 8192, 32768]
        else:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            seq_lens = [1024, 4096, 8192, 16384, 32768, 65536]
        
        run_sweep_benchmark(
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            page_size=args.page_size,
        )


if __name__ == "__main__":
    main()
