#!/usr/bin/env python3
"""
MiniMax-M2.5 decode attention kernel micro-benchmark with baselines.

This script compares multiple attention backends for MiniMax-M2.5 decode:
1. PyTorch SDPA (baseline)
2. FlashInfer (BatchDecodeWithPagedKVCacheWrapper)
3. FlashAttention4 (SM100 only - Blackwell B200/B300/GB200)
4. Theoretical optimal - hardware peak

Hardware support:
- FA4: SM100 (Blackwell B200/B300/GB200) only
- FlashInfer: SM80+ (all modern NVIDIA GPUs)

Usage:
    python scripts/bench_minimax25_decode_micro.py --batch-size 64 --seq-len 8192
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
# B200: 192GB HBM3e @ 8 TB/s peak (unidirectional)
B200_PEAK_BW_TB_S = 8.0


@dataclass
class BenchResult:
    batch_size: int
    seq_len: int
    provider: str
    latency_us: float
    traffic_gb: float
    bandwidth_gb_s: float


def get_device_capability() -> Tuple[int, int]:
    return torch.cuda.get_device_capability()


def is_sm100() -> bool:
    major, _ = get_device_capability()
    return major == 10


def check_flashinfer() -> bool:
    try:
        import flashinfer
        return True
    except ImportError:
        return False


def check_trtllm_gen() -> bool:
    """TRTLLM-GEN kernel: production backend for SM100 (fmhaSm100fKernel)."""
    try:
        import flashinfer.decode
        return hasattr(flashinfer.decode, "trtllm_batch_decode_with_kv_cache")
    except ImportError:
        return False


def check_fa4() -> bool:
    if not is_sm100():
        return False
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func
        return True
    except ImportError:
        return False


def create_decode_inputs(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create inputs for decode phase."""
    q = torch.randn(batch_size, 1, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    return q, k, v


def attn_pytorch_sdpa(q, k, v):
    """PyTorch SDPA baseline with GQA expansion."""
    batch_size = q.shape[0]
    seq_len_k = k.shape[1]
    
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len_k, NUM_Q_HEADS, HEAD_DIM
    )
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len_k, NUM_Q_HEADS, HEAD_DIM
    )
    
    q_t = q.transpose(1, 2)
    k_t = k_exp.transpose(1, 2)
    v_t = v_exp.transpose(1, 2)
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.transpose(1, 2)


def create_flashinfer_wrapper(batch_size: int, seq_len: int, dtype: torch.dtype, device: str):
    """Create and configure FlashInfer decode wrapper."""
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    
    page_size = 16
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    
    # Allocate workspace buffer (128 MB)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    
    # Create wrapper
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=False,
    )
    
    # Create page indices
    page_indices = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages_per_seq
    )
    
    # Indptr: [0, num_pages_per_seq, 2*num_pages_per_seq, ...]
    indptr = torch.arange(0, total_pages + 1, num_pages_per_seq, dtype=torch.int32, device=device)
    
    # Flatten page indices
    indices = page_indices.flatten()
    
    # Last page lengths
    last_page_len = torch.full((batch_size,), seq_len % page_size or page_size, dtype=torch.int32, device=device)
    
    # Plan - FlashInfer handles GQA automatically via num_qo_heads and num_kv_heads
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        pos_encoding_mode="NONE",
        data_type=dtype,
    )
    
    return wrapper


def attn_flashinfer_decode(q, k, v, wrapper=None):
    """FlashInfer decode using BatchDecodeWithPagedKVCacheWrapper."""
    try:
        batch_size = q.shape[0]
        seq_len = k.shape[1]
        
        if wrapper is None:
            wrapper = create_flashinfer_wrapper(batch_size, seq_len, q.dtype, q.device)
        
        page_size = 16
        num_pages = (seq_len + page_size - 1) // page_size
        
        # Pad KV to full pages
        padded_len = num_pages * page_size
        if padded_len > seq_len:
            k_padded = F.pad(k, (0, 0, 0, 0, 0, padded_len - seq_len))
            v_padded = F.pad(v, (0, 0, 0, 0, 0, padded_len - seq_len))
        else:
            k_padded, v_padded = k, v
        
        # Reshape to paged format
        k_pages = k_padded.reshape(batch_size, num_pages, page_size, NUM_KV_HEADS, HEAD_DIM)
        v_pages = v_padded.reshape(batch_size, num_pages, page_size, NUM_KV_HEADS, HEAD_DIM)
        
        # Flatten to [total_pages, page_size, H_KV, D]
        k_cache = k_pages.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM)
        v_cache = v_pages.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM)
        
        paged_kv_cache = (k_cache, v_cache)
        
        # Forward - q needs [B, H_Q, D]
        q_input = q.squeeze(1)
        out = wrapper.forward(q_input, paged_kv_cache, sm_scale=1.0 / (HEAD_DIM ** 0.5))
        
        return out.unsqueeze(1)
    except Exception as e:
        print(f"FlashInfer error: {e}")
        return attn_pytorch_sdpa(q, k, v)


def attn_fa4_decode(q, k_cache, v_cache):
    """FlashAttention4 decode using flash_attn_varlen_func."""
    from flash_attn.cute.interface import flash_attn_varlen_func
    
    batch_size = q.shape[0]
    seq_len = k_cache.shape[1]
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=q.device)
    # [0, seq_len, 2*seq_len, ..., batch_size*seq_len] - shape: (batch_size+1,)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device)
    
    # Reshape for varlen
    q_input = q.squeeze(1)  # [B, H_Q, D]
    k_input = k_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)  # [B*S, H_KV, D]
    v_input = v_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    
    out, lse = flash_attn_varlen_func(
        q=q_input,
        k=k_input,
        v=v_input,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=seq_len,
        causal=True,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5),
    )
    
    return out.unsqueeze(1)


def create_trtllm_paged_kv_cache(k, v, page_size=16):
    """Convert dense KV [B, S, H, D] -> paged format [num_pages, H, page_size, D] for TRTLLM-GEN."""
    batch_size = k.shape[0]
    seq_len = k.shape[1]
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    padded_len = num_pages_per_seq * page_size

    if padded_len > seq_len:
        k = F.pad(k, (0, 0, 0, 0, 0, padded_len - seq_len))
        v = F.pad(v, (0, 0, 0, 0, 0, padded_len - seq_len))

    # [B, S_pad, H, D] -> [B, num_pages, page_size, H, D] -> [B*num_pages, H, page_size, D]
    k_pages = k.reshape(batch_size, num_pages_per_seq, page_size, NUM_KV_HEADS, HEAD_DIM)
    v_pages = v.reshape(batch_size, num_pages_per_seq, page_size, NUM_KV_HEADS, HEAD_DIM)
    # permute: [B*P, H, page_size, D]
    k_cache = k_pages.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    v_cache = v_pages.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    return k_cache, v_cache, num_pages_per_seq


def attn_trtllm_gen_decode(q, k, v, trtllm_state=None):
    """TRTLLM-GEN decode using flashinfer.decode.trtllm_batch_decode_with_kv_cache.
    
    This is the production kernel: fmhaSm100fKernel (SM100/Blackwell only).
    KV cache layout: [num_pages, num_kv_heads, page_size, head_dim]
    """
    import flashinfer.decode

    batch_size = q.shape[0]
    seq_len = k.shape[1]
    page_size = 16

    if trtllm_state is None:
        workspace_buffer = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        k_cache, v_cache, num_pages_per_seq = create_trtllm_paged_kv_cache(k, v, page_size)
    else:
        workspace_buffer, k_cache, v_cache, num_pages_per_seq = trtllm_state

    # block_tables: [B, num_pages_per_seq] - page indices for each sequence
    # Sequences are packed contiguously: seq i uses pages [i*num_pages_per_seq, (i+1)*num_pages_per_seq)
    block_tables = torch.arange(
        batch_size * num_pages_per_seq, dtype=torch.int32, device=q.device
    ).reshape(batch_size, num_pages_per_seq)

    # seq_lens: [B] - actual sequence lengths
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=q.device)

    kv_cache = (k_cache, v_cache)

    # q: [B, H_Q, D] for trtllm kernel
    q_input = q.squeeze(1).contiguous()  # [B, H_Q, D]

    scale = 1.0 / (HEAD_DIM ** 0.5)
    o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q_input,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        window_left=-1,
        out_dtype=q.dtype,
    )

    return o.unsqueeze(1)  # [B, 1, H_Q, D]


def create_trtllm_state(batch_size: int, seq_len: int, dtype: torch.dtype, device: str):
    """Pre-allocate TRTLLM workspace and KV cache for benchmarking."""
    workspace_buffer = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k_cache, v_cache, num_pages_per_seq = create_trtllm_paged_kv_cache(k, v)
    return workspace_buffer, k_cache, v_cache, num_pages_per_seq


def calculate_theoretical_latency(batch_size: int, seq_len: int, dtype: torch.dtype = torch.bfloat16):
    element_size = 2
    q_bytes = batch_size * 1 * NUM_Q_HEADS * HEAD_DIM * element_size
    kv_bytes = 2 * batch_size * seq_len * NUM_KV_HEADS * HEAD_DIM * element_size
    o_bytes = batch_size * 1 * NUM_Q_HEADS * HEAD_DIM * element_size
    total_bytes = q_bytes + kv_bytes + o_bytes
    
    traffic_gb = total_bytes / 1e9
    theoretical_latency_us = traffic_gb / B200_PEAK_BW_TB_S / 1e3 * 1e6  # us
    
    return theoretical_latency_us, traffic_gb


def bench_provider(
    provider_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_iters: int = 20,
    bench_iters: int = 100,
    wrapper=None,
) -> float:
    """Benchmark provider, returns latency in microseconds."""
    # Warmup
    for _ in range(warmup_iters):
        if wrapper is not None:
            _ = provider_fn(q, k, v, wrapper)
        else:
            _ = provider_fn(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(bench_iters):
        if wrapper is not None:
            _ = provider_fn(q, k, v, wrapper)
        else:
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
        bandwidth_gb_s=traffic_gb / (theoretical_us * 1e-6),  # GB/s
    ))
    
    # PyTorch SDPA (always available)
    providers = {
        "PyTorch SDPA": (attn_pytorch_sdpa, None),
    }
    
    # FlashInfer with pre-created wrapper
    flashinfer_wrapper = None
    if check_flashinfer():
        try:
            flashinfer_wrapper = create_flashinfer_wrapper(batch_size, seq_len, dtype, 'cuda')
            providers["FlashInfer"] = (
                lambda q, k, v, w=flashinfer_wrapper: attn_flashinfer_decode(q, k, v, w),
                flashinfer_wrapper
            )
        except Exception as e:
            print(f"  FlashInfer wrapper creation failed: {e}")
    
    # FA4 (SM100 only)
    if check_fa4():
        providers["FA4"] = (attn_fa4_decode, None)

    # TRTLLM-GEN (production backend for SM100: fmhaSm100fKernel)
    if check_trtllm_gen():
        try:
            trtllm_state = create_trtllm_state(batch_size, seq_len, dtype, 'cuda')
            providers["TRTLLM-GEN"] = (
                lambda q, k, v, s=trtllm_state: attn_trtllm_gen_decode(q, k, v, s),
                trtllm_state,
            )
        except Exception as e:
            print(f"  TRTLLM-GEN state creation failed: {e}")
    
    for name, (fn, wrapper) in providers.items():
        try:
            latency_us = bench_provider(fn, q, k, v, warmup_iters, bench_iters, wrapper)
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
        description="MiniMax-M2.5 decode attention micro-benchmark"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sweep"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    major, minor = get_device_capability()
    
    print("=" * 80)
    print("MiniMax-M2.5 Decode Attention Benchmark with Baselines")
    print(f"Config: TP={TP_SIZE}, Per-GPU: {NUM_Q_HEADS} Q heads, {NUM_KV_HEADS} KV heads (GQA ratio={GQA_RATIO})")
    print(f"Hardware: SM{major}{minor}")
    print("=" * 80)
    
    # Check available providers
    print("\nAvailable providers:")
    print(f"  PyTorch SDPA: ✓ (always available)")
    print(f"  FlashInfer: {'✓' if check_flashinfer() else '✗'}")
    print(f"  TRTLLM-GEN: {'✓ (SM100 production kernel)' if check_trtllm_gen() else '✗'}")
    print(f"  FA4: {'✓ (SM100 Blackwell)' if check_fa4() else '✗' + (' (requires SM100)' if major != 10 else '')}")
    
    if args.mode == "single":
        print(f"\nRunning: batch_size={args.batch_size}, seq_len={args.seq_len}")
        results = run_single_benchmark(args.batch_size, args.seq_len)
        print_comparison_table(results)
        
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
            
            total_attn_ms = best.latency_us * NUM_LAYERS / 1000
            print(f"\n  Total attention time ({NUM_LAYERS} layers): {total_attn_ms:.2f} ms")
            
            print(f"\n  Attention share at 50ms decode budget: {total_attn_ms/50*100:.1f}%")
            if total_attn_ms > 20:
                print("  ⚠️  Attention overhead is HIGH (>40% of 50ms budget)")
            else:
                print("  ✓ Attention overhead is within typical bounds")
    
    else:
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
        
        print("\n" + "=" * 80)
        print("Final Summary:")
        print("=" * 80)
        
        providers = set(r.provider for r in all_results if r.provider != "Theoretical")
        for provider in providers:
            provider_results = [r for r in all_results if r.provider == provider]
            if provider_results:
                avg_latency = sum(r.latency_us for r in provider_results) / len(provider_results)
                print(f"{provider}: avg latency = {avg_latency:.2f} us")


if __name__ == "__main__":
    main()
