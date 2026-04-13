#!/usr/bin/env python3
"""
MiniMax-M2.5 decode attention kernel micro-benchmark with baselines.

This script compares multiple attention backends for MiniMax-M2.5 decode:
1. PyTorch SDPA (baseline)
2. FlashInfer (if available) - SGLang default
3. FlashAttention3 (SM90 only - Hopper H100/H200)
4. FlashAttention4 (SM100 only - Blackwell B200/B300/GB200)
5. Theoretical optimal - hardware peak

Usage:
    python scripts/bench_minimax25_decode_micro.py --batch-size 64 --seq-len 8192
    python scripts/bench_minimax25_decode_micro.py --mode sweep --quick

Hardware support:
- FA3: SM90 (Hopper H100/H200) only
- FA4: SM100 (Blackwell B200/B300/GB200) only  
- FlashInfer: SM80+ (all modern NVIDIA GPUs)
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

# ============== MiniMax-M2.5 Config ==============
NUM_Q_HEADS_TOTAL = 48
NUM_KV_HEADS_TOTAL = 8
HEAD_DIM = 128
NUM_LAYERS = 62

# Per-GPU config (TP=4)
TP_SIZE = 4
NUM_Q_HEADS = NUM_Q_HEADS_TOTAL // TP_SIZE  # 12
NUM_KV_HEADS = NUM_KV_HEADS_TOTAL // TP_SIZE  # 2
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 6

# Hardware specs for theoretical baseline
B200_PEAK_BW_TB_S = 6.0


@dataclass
class BenchResult:
    batch_size: int
    seq_len: int
    provider: str
    latency_us: float
    traffic_gb: float
    bandwidth_gb_s: float


def get_device_capability() -> tuple:
    """Get CUDA device capability (major, minor)."""
    return torch.cuda.get_device_capability()


def is_sm90() -> bool:
    """Check if running on SM90 (Hopper H100/H200)."""
    major, _ = get_device_capability()
    return major == 9


def is_sm100() -> bool:
    """Check if running on SM100 (Blackwell B200/B300/GB200)."""
    major, _ = get_device_capability()
    return major == 10


def check_flashinfer() -> bool:
    """Check if FlashInfer is available."""
    try:
        import flashinfer
        return True
    except ImportError:
        return False


def check_fa3() -> bool:
    """Check if FlashAttention3 is available (SM90 only)."""
    if not is_sm90():
        return False
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        pass
    try:
        from flash_attn_interface import flash_attn_func
        return True
    except ImportError:
        pass
    return False


def check_fa4() -> bool:
    """Check if FlashAttention4 is available (SM100 only)."""
    if not is_sm100():
        return False
    try:
        from flash_attn_4 import flash_attn_func
        return True
    except ImportError:
        pass
    try:
        # Alternative import path
        import flash_attn_4
        return hasattr(flash_attn_4, 'flash_attn_func')
    except ImportError:
        pass
    return False


def create_decode_inputs(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create inputs simulating decode phase."""
    q = torch.randn(batch_size, 1, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    return q, k, v


def attn_pytorch_sdpa(q, k, v):
    """PyTorch SDPA baseline with GQA expansion."""
    batch_size = q.shape[0]
    seq_len_k = k.shape[1]
    
    # Expand K/V for GQA
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len_k, NUM_Q_HEADS, HEAD_DIM
    )
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len_k, NUM_Q_HEADS, HEAD_DIM
    )
    
    # [B, H, S, D]
    q_t = q.transpose(1, 2)  # [B, H_Q, 1, D]
    k_t = k_exp.transpose(1, 2)  # [B, H_Q, S, D]
    v_t = v_exp.transpose(1, 2)  # [B, H_Q, S, D]
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.transpose(1, 2)  # [B, 1, H_Q, D]


def attn_flashinfer_decode(q, k, v):
    """FlashInfer decode attention."""
    try:
        import flashinfer
        
        batch_size = q.shape[0]
        seq_len = k.shape[1]
        
        # FlashInfer uses paged KV cache
        page_size = 16
        num_pages = (seq_len + page_size - 1) // page_size
        total_pages = batch_size * num_pages
        
        page_table = torch.arange(total_pages, dtype=torch.int32, device=q.device).reshape(
            batch_size, num_pages
        )
        
        k_paged = k.reshape(batch_size, num_pages, page_size, NUM_KV_HEADS, HEAD_DIM)
        v_paged = v.reshape(batch_size, num_pages, page_size, NUM_KV_HEADS, HEAD_DIM)
        k_flat = k_paged.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM)
        v_flat = v_paged.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM)
        
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=q.device)
        
        # Use batch decode API
        out = flashinfer.batch_decode_with_padded_kv_cache(
            q.squeeze(1),  # [B, H_Q, D]
            k_flat,
            v_flat,
            kv_lengths=seq_lens,
            num_qo_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
        )
        return out.unsqueeze(1)  # [B, 1, H_Q, D]
    except Exception as e:
        print(f"FlashInfer error: {e}, falling back to SDPA")
        return attn_pytorch_sdpa(q, k, v)


def attn_fa3(q, k, v):
    """FlashAttention3 decode (SM90 Hopper only)."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        from flash_attn_interface import flash_attn_func
    
    batch_size = q.shape[0]
    seq_len = k.shape[1]
    
    # Expand K/V for GQA
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM
    )
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM
    )
    
    # flash_attn_func: [B, S, H, D]
    out, _, _ = flash_attn_func(
        q, k_exp, v_exp,
        causal=True,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5),
    )
    return out


def attn_fa4(q, k, v):
    """FlashAttention4 decode (SM100 Blackwell only)."""
    try:
        from flash_attn_4 import flash_attn_func
    except ImportError:
        raise ImportError("flash_attn_4 not installed. Install with: pip install flash-attn-4")
    
    batch_size = q.shape[0]
    seq_len = k.shape[1]
    
    # FA4 handles GQA internally if num_heads differ
    # But we still need to expand for the interface
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM
    )
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len, NUM_Q_HEADS, HEAD_DIM
    )
    
    out, lse = flash_attn_func(
        q, k_exp, v_exp,
        causal=True,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5),
    )
    return out


def calculate_theoretical_latency(batch_size: int, seq_len: int, dtype: torch.dtype = torch.bfloat16):
    """Calculate theoretical minimum latency based on memory bandwidth."""
    element_size = 2  # bf16
    
    # Memory traffic (bytes)
    q_bytes = batch_size * 1 * NUM_Q_HEADS * HEAD_DIM * element_size
    kv_bytes = 2 * batch_size * seq_len * NUM_KV_HEADS * HEAD_DIM * element_size
    o_bytes = batch_size * 1 * NUM_Q_HEADS * HEAD_DIM * element_size
    total_bytes = q_bytes + kv_bytes + o_bytes
    
    theoretical_latency_us = total_bytes / (B200_PEAK_BW_TB_S * 1e12) * 1e6
    traffic_gb = total_bytes / 1e9
    
    return theoretical_latency_us, traffic_gb


def bench_provider(
    provider_fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_iters: int = 20,
    bench_iters: int = 100,
) -> float:
    """Benchmark a single provider, returns latency in microseconds."""
    # Warmup
    for _ in range(warmup_iters):
        _ = provider_fn(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(bench_iters):
        _ = provider_fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / bench_iters
    return latency_ms * 1000


def run_single_benchmark(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 20,
    bench_iters: int = 100,
) -> List[BenchResult]:
    """Run benchmark for all available providers."""
    results = []
    
    major, minor = get_device_capability()
    print(f"  GPU: SM{major}{minor}")
    
    q, k, v = create_decode_inputs(batch_size, seq_len, dtype)
    theoretical_us, traffic_gb = calculate_theoretical_latency(batch_size, seq_len, dtype)
    
    # Theoretical baseline
    results.append(BenchResult(
        batch_size=batch_size,
        seq_len=seq_len,
        provider="Theoretical",
        latency_us=theoretical_us,
        traffic_gb=traffic_gb,
        bandwidth_gb_s=traffic_gb / (theoretical_us * 1e-6) / 1e9,
    ))
    
    # PyTorch SDPA (always available)
    providers = {
        "PyTorch SDPA": attn_pytorch_sdpa,
    }
    
    # FlashInfer
    if check_flashinfer():
        providers["FlashInfer"] = attn_flashinfer_decode
    
    # FA3 (SM90 only)
    if check_fa3():
        providers["FA3"] = attn_fa3
    
    # FA4 (SM100 only)
    if check_fa4():
        providers["FA4"] = attn_fa4
    
    for name, fn in providers.items():
        try:
            latency_us = bench_provider(fn, q, k, v, warmup_iters, bench_iters)
            bandwidth_gb_s = traffic_gb / (latency_us * 1e-6) / 1e9
            results.append(BenchResult(
                batch_size=batch_size,
                seq_len=seq_len,
                provider=name,
                latency_us=latency_us,
                traffic_gb=traffic_gb,
                bandwidth_gb_s=bandwidth_gb_s,
            ))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")
    
    return results


def print_comparison_table(results: List[BenchResult]):
    """Print comparison table."""
    print(f"\n{'Provider':>15} | {'Latency (us)':>12} | {'BW (GB/s)':>10} | {'vs Theoretical':>15}")
    print("-" * 65)
    
    theoretical = next(r for r in results if r.provider == "Theoretical")
    
    for r in sorted(results, key=lambda x: x.latency_us):
        if r.provider == "Theoretical":
            print(f"{r.provider:>15} | {r.latency_us:>12.2f} | {r.bandwidth_gb_s:>10.1f} | {'1.00x (baseline)':>15}")
        else:
            ratio = r.latency_us / theoretical.latency_us
            print(f"{r.provider:>15} | {r.latency_us:>12.2f} | {r.bandwidth_gb_s:>10.1f} | {ratio:>14.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax-M2.5 decode attention micro-benchmark with baselines"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sweep"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    major, minor = get_device_capability()
    
    print("=" * 80)
    print("MiniMax-M2.5 Decode Attention Benchmark with Baselines")
    print(f"Config: TP={TP_SIZE}, Per-GPU: {NUM_Q_HEADS} Q heads, {NUM_KV_HEADS} KV heads")
    print(f"Hardware: SM{major}{minor}")
    print("=" * 80)
    
    # Check available providers
    print("\nAvailable providers:")
    print(f"  PyTorch SDPA: ✓ (always available)")
    print(f"  FlashInfer: {'✓' if check_flashinfer() else '✗'}")
    print(f"  FA3: {'✓ (SM90 Hopper)' if check_fa3() else '✗' + (' (requires SM90, current SM' + str(major) + ')' if major != 9 else '')}")
    print(f"  FA4: {'✓ (SM100 Blackwell)' if check_fa4() else '✗' + (' (requires SM100, current SM' + str(major) + ')' if major != 10 else '')}")
    
    if args.mode == "single":
        print(f"\nRunning: batch_size={args.batch_size}, seq_len={args.seq_len}")
        results = run_single_benchmark(args.batch_size, args.seq_len)
        print_comparison_table(results)
        
        # Summary
        print(f"\n{'='*80}")
        print("Summary:")
        
        best = min((r for r in results if r.provider != "Theoretical"), 
                   key=lambda x: x.latency_us, default=None)
        
        if best:
            theoretical = next(r for r in results if r.provider == "Theoretical")
            efficiency = theoretical.latency_us / best.latency_us * 100
            print(f"  Best provider: {best.provider}")
            print(f"  Latency: {best.latency_us:.2f} us/layer")
            print(f"  Bandwidth efficiency: {efficiency:.1f}% of theoretical peak")
            
            # Total attention time
            total_attn_ms = best.latency_us * NUM_LAYERS / 1000
            print(f"\n  Total attention time ({NUM_LAYERS} layers): {total_attn_ms:.2f} ms")
            
            print(f"\n  Attention share at 50ms decode budget: {total_attn_ms/50*100:.1f}%")
            if total_attn_ms > 20:
                print("  ⚠️  Attention overhead is HIGH (>40% of 50ms budget)")
            else:
                print("  ✓ Attention overhead is within typical bounds")
    
    else:
        # Sweep mode
        if args.quick:
            batch_sizes = [1, 8, 32, 64, 128]
            seq_lens = [1024, 4096, 8192, 32768]
        else:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
        
        print(f"\nSweeping {len(batch_sizes)} batch sizes x {len(seq_lens)} sequence lengths...")
        
        all_results = []
        for bs in batch_sizes:
            for sl in seq_lens:
                print(f"\nbatch_size={bs}, seq_len={sl}:")
                try:
                    results = run_single_benchmark(bs, sl)
                    print_comparison_table(results)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  Error: {e}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("Final Summary:")
        print("=" * 80)
        
        # Group by provider
        providers = set(r.provider for r in all_results if r.provider != "Theoretical")
        for provider in providers:
            provider_results = [r for r in all_results if r.provider == provider]
            if provider_results:
                avg_latency = sum(r.latency_us for r in provider_results) / len(provider_results)
                print(f"{provider}: avg latency = {avg_latency:.2f} us")


if __name__ == "__main__":
    main()
