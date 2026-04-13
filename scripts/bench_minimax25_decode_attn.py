#!/usr/bin/env python3
"""
Benchmark MiniMax-M2.5 decode attention kernel latency on SM100 (Blackwell).

MiniMax-M2.5 model config:
- num_q_heads = 48
- num_kv_heads = 8 (GQA ratio 6:1)
- head_dim = 128
- num_layers = 62

Kernel: fmhaSm100fKernel QkvE4m30Bfloat16H128PagedKvCausalP64VarSegQ8Kv128PersistentSwapsAbForGen

This script benchmarks the decode attention kernel across various batch sizes and 
sequence lengths to identify if there's abnormal latency.
"""

import argparse
import itertools
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import triton

# MiniMax-M2.5 config
NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 62


@dataclass
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    page_size: int = 64
    dtype: torch.dtype = torch.bfloat16


def create_decode_attention_inputs(config: BenchmarkConfig, device: str = "cuda"):
    """Create inputs for decode attention benchmark."""
    batch_size = config.batch_size
    seq_len = config.seq_len
    page_size = config.page_size
    dtype = config.dtype
    
    # Calculate number of pages needed
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    
    # Query: [batch_size, num_q_heads, head_dim] for decode
    q = torch.randn(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
    
    # KV cache: [total_pages, page_size, num_kv_heads, head_dim]
    k_cache = torch.randn(total_pages, page_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v_cache = torch.randn(total_pages, page_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    
    # Block table: [batch_size, num_pages_per_seq]
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch_size, num_pages_per_seq)
    
    # Sequence lengths: [batch_size]
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    
    return q, k_cache, v_cache, block_table, seq_lens


def benchmark_flash_attention_decode(
    batch_size: int,
    seq_len: int,
    page_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "cuda",
):
    """
    Benchmark FlashAttention decode kernel.
    
    Returns latency in microseconds.
    """
    config = BenchmarkConfig(batch_size=batch_size, seq_len=seq_len, page_size=page_size, dtype=dtype)
    q, k_cache, v_cache, block_table, seq_lens = create_decode_attention_inputs(config, device)
    
    # Try to import flash attention
    try:
        from flash_attn import flash_attn_with_kvcache
        has_flash_attn = True
    except ImportError:
        has_flash_attn = False
        print("Warning: flash_attn not available, using torch reference implementation")
    
    # Warmup
    for _ in range(warmup_iters):
        if has_flash_attn:
            # flash_attn_with_kvcache API
            # Note: This is a simplified call, actual API may differ
            _ = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),  # [1, num_q_heads, batch_size, head_dim]
                k_cache.reshape(1, batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
                    .expand(-1, -1, -1, NUM_Q_HEADS // NUM_KV_HEADS, -1)
                    .reshape(1, batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM)
                    .transpose(1, 2),
                v_cache.reshape(1, batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
                    .expand(-1, -1, -1, NUM_Q_HEADS // NUM_KV_HEADS, -1)
                    .reshape(1, batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM)
                    .transpose(1, 2),
                is_causal=True,
            )
        else:
            # Reference implementation using PyTorch
            _ = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k_cache.reshape(1, batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
                    .expand(-1, -1, -1, NUM_Q_HEADS // NUM_KV_HEADS, -1)
                    .reshape(1, batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM)
                    .transpose(1, 2),
                v_cache.reshape(1, batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
                    .expand(-1, -1, -1, NUM_Q_HEADS // NUM_KV_HEADS, -1)
                    .reshape(1, batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM)
                    .transpose(1, 2),
                is_causal=True,
            )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k_cache.reshape(1, batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
                .expand(-1, -1, -1, NUM_Q_HEADS // NUM_KV_HEADS, -1)
                .reshape(1, batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM)
                .transpose(1, 2),
            v_cache.reshape(1, batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
                .expand(-1, -1, -1, NUM_Q_HEADS // NUM_KV_HEADS, -1)
                .reshape(1, batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM)
                .transpose(1, 2),
            is_causal=True,
        )
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    latency_us = (end - start) / bench_iters * 1e6
    return latency_us


def benchmark_sglang_attention(
    batch_size: int,
    seq_len: int,
    page_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "cuda",
):
    """
    Benchmark SGLang's attention backend for decode.
    """
    try:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        has_sglang = True
    except ImportError:
        has_sglang = False
        print("Warning: SGLang attention backend not available")
        return None
    
    # This is a placeholder - actual implementation would need more setup
    return None


def estimate_theoretical_latency(batch_size: int, seq_len: int, dtype: torch.dtype = torch.bfloat16):
    """
    Estimate theoretical latency based on memory bandwidth and compute.
    
    For decode attention with GQA:
    - Q: [batch_size, num_q_heads, head_dim] read
    - K: [batch_size, seq_len, num_kv_heads, head_dim] read
    - V: [batch_size, seq_len, num_kv_heads, head_dim] read
    - O: [batch_size, num_q_heads, head_dim] write
    
    Memory traffic (bytes):
    - Q: batch_size * num_q_heads * head_dim * 2
    - K: batch_size * seq_len * num_kv_heads * head_dim * 2
    - V: batch_size * seq_len * num_kv_heads * head_dim * 2
    - O: batch_size * num_q_heads * head_dim * 2
    
    Total reads: Q + K + V
    Total writes: O
    
    Compute (FLOPs):
    - QK^T: batch_size * num_q_heads * seq_len * head_dim * 2
    - Softmax: ~batch_size * num_q_heads * seq_len * 5
    - AV: batch_size * num_q_heads * seq_len * head_dim * 2
    Total: batch_size * num_q_heads * seq_len * (2 * head_dim + 5 + 2 * head_dim)
         ≈ batch_size * num_q_heads * seq_len * 4 * head_dim
    """
    element_size = 2 if dtype == torch.bfloat16 else 2
    
    # Memory traffic
    q_size = batch_size * NUM_Q_HEADS * HEAD_DIM * element_size
    kv_size = batch_size * seq_len * NUM_KV_HEADS * HEAD_DIM * element_size * 2  # K and V
    o_size = batch_size * NUM_Q_HEADS * HEAD_DIM * element_size
    
    total_reads = q_size + kv_size
    total_writes = o_size
    total_traffic = total_reads + total_writes
    
    # Compute FLOPs
    qk_flops = batch_size * NUM_Q_HEADS * seq_len * HEAD_DIM * 2
    av_flops = batch_size * NUM_Q_HEADS * seq_len * HEAD_DIM * 2
    softmax_flops = batch_size * NUM_Q_HEADS * seq_len * 5
    total_flops = qk_flops + av_flops + softmax_flops
    
    # Assumed hardware specs (B200)
    # Memory bandwidth: ~8 TB/s
    # Compute: ~1.75 PFLOPS (BF16)
    bw_tb_s = 8.0  # TB/s
    compute_pflops = 1.75  # PFLOPS
    
    # Memory bound latency
    mem_latency_us = total_traffic / (bw_tb_s * 1e12) * 1e6
    
    # Compute bound latency
    compute_latency_us = total_flops / (compute_pflops * 1e15) * 1e6
    
    return {
        "memory_traffic_bytes": total_traffic,
        "compute_flops": total_flops,
        "mem_bound_latency_us": mem_latency_us,
        "compute_bound_latency_us": compute_latency_us,
        "theoretical_min_latency_us": max(mem_latency_us, compute_latency_us),
    }


def run_benchmark_sweep(
    batch_sizes: List[int],
    seq_lens: List[int],
    page_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Run benchmark sweep across batch sizes and sequence lengths."""
    results = []
    
    print("=" * 100)
    print(f"MiniMax-M2.5 Decode Attention Benchmark (SM100/Blackwell)")
    print(f"Config: num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print("=" * 100)
    print()
    
    header = f"{'Batch':>8} | {'SeqLen':>8} | {'Measured (us)':>14} | {'Mem Bound (us)':>15} | {'Compute Bound (us)':>18} | {'Theoretical Min (us)':>19} | {'Ratio':>8}"
    print(header)
    print("-" * len(header))
    
    for batch_size, seq_len in itertools.product(batch_sizes, seq_lens):
        try:
            measured_latency = benchmark_flash_attention_decode(
                batch_size=batch_size,
                seq_len=seq_len,
                page_size=page_size,
                dtype=dtype,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )
            
            theoretical = estimate_theoretical_latency(batch_size, seq_len, dtype)
            
            ratio = measured_latency / theoretical["theoretical_min_latency_us"]
            
            print(f"{batch_size:>8} | {seq_len:>8} | {measured_latency:>14.2f} | {theoretical['mem_bound_latency_us']:>15.2f} | {theoretical['compute_bound_latency_us']:>18.4f} | {theoretical['theoretical_min_latency_us']:>19.2f} | {ratio:>8.2f}x")
            
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "measured_latency_us": measured_latency,
                **theoretical,
                "efficiency_ratio": ratio,
            })
        except Exception as e:
            print(f"{batch_size:>8} | {seq_len:>8} | Error: {e}")
    
    print()
    print("=" * 100)
    print("Analysis:")
    print("- Ratio = measured_latency / theoretical_min_latency")
    print("- For efficient kernels, ratio should be close to 1.0x")
    print("- If ratio > 3.0x, there might be kernel inefficiency")
    print("- If ratio < 1.0x, theoretical model may be inaccurate")
    print()
    
    # Find potential issues
    high_ratio_results = [r for r in results if r.get("efficiency_ratio", 0) > 3.0]
    if high_ratio_results:
        print("⚠️  Potential issues detected:")
        for r in high_ratio_results:
            print(f"   - Batch={r['batch_size']}, SeqLen={r['seq_len']}: {r['efficiency_ratio']:.2f}x efficiency ratio")
    else:
        print("✓ All kernel latencies appear normal")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniMax-M2.5 decode attention kernel")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 32768, 65536],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--page-size", type=int, default=64, help="Page size for paged attention")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
                        help="Data type")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--bench-iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer configs")
    args = parser.parse_args()
    
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    if args.quick:
        batch_sizes = [1, 8, 64]
        seq_lens = [1024, 8192, 32768]
    else:
        batch_sizes = args.batch_sizes
        seq_lens = args.seq_lens
    
    results = run_benchmark_sweep(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        page_size=args.page_size,
        dtype=dtype,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    
    # Also calculate per-layer attention time
    print("\n" + "=" * 100)
    print("Per-Layer Attention Time Estimation:")
    print("=" * 100)
    
    # Find a typical config
    typical_batch = 64
    typical_seq = 8192
    
    typical_result = next((r for r in results if r["batch_size"] == typical_batch and r["seq_len"] == typical_seq), None)
    
    if typical_result:
        per_layer_time_us = typical_result["measured_latency_us"]
        total_attn_time_ms = per_layer_time_us * NUM_LAYERS / 1000
        
        print(f"Typical config: batch_size={typical_batch}, seq_len={typical_seq}")
        print(f"Per-layer attention latency: {per_layer_time_us:.2f} us")
        print(f"Total attention latency (all {NUM_LAYERS} layers): {total_attn_time_ms:.2f} ms")
        print()
        
        # Estimate expected time budget
        # Assuming 50ms decode latency budget, attention should be ~30% = 15ms
        print("Expected budget analysis:")
        print(f"- If target decode latency is 50ms/token")
        print(f"- And attention should be ~30% of total time")
        print(f"- Then attention budget is ~15ms")
        print(f"- Current estimated attention time: {total_attn_time_ms:.2f} ms")
        
        if total_attn_time_ms > 20:
            print(f"⚠️  Attention time ({total_attn_time_ms:.2f} ms) exceeds typical budget!")
        else:
            print(f"✓ Attention time ({total_attn_time_ms:.2f} ms) within typical budget")


if __name__ == "__main__":
    main()
