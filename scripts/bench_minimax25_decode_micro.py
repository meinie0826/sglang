#!/usr/bin/env python3
"""
MiniMax-M2.5 decode attention kernel micro-benchmark with baselines.

Production kernel: fmhaSm100fKernel QkvE4m30Bfloat16H128PagedKvCausalP64VarSegQ8Kv128PersistentSwapsAbForGen
  - KV dtype: FP8 E4M3  (--kv-dtype fp8)
  - Q dtype:  BF16
  - head_dim: 128
  - Page size: 64       (--page-size 64)
  - Q tokens per req: 8 (--q-len 8, EAGLE speculative decoding)
  - Causal, Paged KV

Providers:
1. PyTorch SDPA (baseline, BF16 only)
2. TRTLLM-GEN (production: flashinfer.decode.trtllm_batch_decode_with_kv_cache)
3. FlashAttention4 (SM100 only)
4. Theoretical optimal (memory-bound roofline)

Usage:
    # Simulate production (FP8 KV, q_len=8, page_size=64):
    python scripts/bench_minimax25_decode_micro.py --batch-size 64 --seq-len 8192 \\
        --kv-dtype fp8 --q-len 8 --page-size 64

    # Sweep seq_len 8k~32k with production config:
    python scripts/bench_minimax25_decode_micro.py --batch-size 64 --mode seq_sweep \\
        --kv-dtype fp8 --q-len 8 --page-size 64
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple

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


def dtype_element_size(dtype: torch.dtype) -> int:
    return {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
        torch.float8_e4m3fn: 1,
    }.get(dtype, 2)


def create_decode_inputs(
    batch_size: int,
    seq_len: int,
    q_len: int = 1,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create inputs for decode/speculative phase.
    
    q_len=1: standard decode
    q_len=8: EAGLE speculative decode (Q8 in kernel name)
    """
    q = torch.randn(batch_size, q_len, NUM_Q_HEADS, HEAD_DIM, dtype=q_dtype, device=device)
    # FP8 doesn't support randn; use random bytes cast instead
    if kv_dtype == torch.float8_e4m3fn:
        k = torch.rand(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM,
                       dtype=torch.float16, device=device).to(kv_dtype)
        v = torch.rand(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM,
                       dtype=torch.float16, device=device).to(kv_dtype)
    else:
        k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=kv_dtype, device=device)
        v = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=kv_dtype, device=device)
    return q, k, v


def attn_pytorch_sdpa(q, k, v):
    """PyTorch SDPA baseline with GQA expansion (BF16 only)."""
    batch_size = q.shape[0]
    q_len = q.shape[1]
    seq_len_k = k.shape[1]

    # Cast to bf16 if needed (SDPA doesn't support fp8)
    q_f = q.to(torch.bfloat16)
    k_f = k.to(torch.bfloat16)
    v_f = v.to(torch.bfloat16)

    k_exp = k_f.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len_k, NUM_Q_HEADS, HEAD_DIM
    )
    v_exp = v_f.unsqueeze(3).expand(-1, -1, -1, GQA_RATIO, -1).reshape(
        batch_size, seq_len_k, NUM_Q_HEADS, HEAD_DIM
    )

    q_t = q_f.transpose(1, 2)   # [B, H_Q, q_len, D]
    k_t = k_exp.transpose(1, 2) # [B, H_Q, S, D]
    v_t = v_exp.transpose(1, 2)

    scale = 1.0 / (HEAD_DIM ** 0.5)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.transpose(1, 2)


def attn_fa4_decode(q, k_cache, v_cache):
    """FlashAttention4 decode/speculative using flash_attn_varlen_func.
    
    Supports arbitrary q_len (decode q_len=1, speculative q_len=8).
    """
    from flash_attn.cute.interface import flash_attn_varlen_func

    batch_size = q.shape[0]
    q_len = q.shape[1]
    seq_len = k_cache.shape[1]

    # FA4 requires bf16; cast fp8 KV
    q_f = q.to(torch.bfloat16)
    k_f = k_cache.to(torch.bfloat16)
    v_f = v_cache.to(torch.bfloat16)

    # cu_seqlens_q: [0, q_len, 2*q_len, ..., B*q_len]
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, q_len,
                                dtype=torch.int32, device=q.device)
    # cu_seqlens_k: [0, seq_len, 2*seq_len, ..., B*seq_len]
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_len, seq_len,
                                dtype=torch.int32, device=q.device)

    q_input = q_f.reshape(-1, NUM_Q_HEADS, HEAD_DIM)    # [B*q_len, H_Q, D]
    k_input = k_f.reshape(-1, NUM_KV_HEADS, HEAD_DIM)   # [B*S, H_KV, D]
    v_input = v_f.reshape(-1, NUM_KV_HEADS, HEAD_DIM)

    out, _ = flash_attn_varlen_func(
        q=q_input,
        k=k_input,
        v=v_input,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_len,
        max_seqlen_k=seq_len,
        causal=True,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5),
    )

    return out.reshape(batch_size, q_len, NUM_Q_HEADS, HEAD_DIM)


def create_trtllm_paged_kv_cache(k, v, page_size=64):
    """Convert dense KV [B, S, H, D] -> paged format [num_pages, H, page_size, D].
    
    Production uses page_size=64 (P64 in kernel name).
    KV layout for trtllm kernel: [num_pages, num_kv_heads, page_size, head_dim]
    """
    batch_size = k.shape[0]
    seq_len = k.shape[1]
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    padded_len = num_pages_per_seq * page_size

    if padded_len > seq_len:
        # fp8 tensors need special handling for pad
        k = torch.cat([k, k.new_zeros(batch_size, padded_len - seq_len, *k.shape[2:])], dim=1)
        v = torch.cat([v, v.new_zeros(batch_size, padded_len - seq_len, *v.shape[2:])], dim=1)

    # [B, S_pad, H, D] -> [B*num_pages, H, page_size, D]
    k_pages = k.reshape(batch_size, num_pages_per_seq, page_size, NUM_KV_HEADS, HEAD_DIM)
    v_pages = v.reshape(batch_size, num_pages_per_seq, page_size, NUM_KV_HEADS, HEAD_DIM)
    k_cache = k_pages.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    v_cache = v_pages.reshape(-1, page_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    return k_cache, v_cache, num_pages_per_seq


def attn_trtllm_gen_decode(q, k, v, trtllm_state=None):
    """TRTLLM-GEN: flashinfer.decode.trtllm_batch_decode_with_kv_cache.
    
    Production kernel: fmhaSm100fKernel
    - Supports FP8 KV cache (E4M3)
    - Supports q_len > 1 via q_len_per_req (EAGLE speculative)
    - KV layout: [num_pages, num_kv_heads, page_size, head_dim]
    """
    import flashinfer.decode

    batch_size = q.shape[0]
    q_len = q.shape[1]
    seq_len = k.shape[1]

    if trtllm_state is None:
        page_size = 64
        workspace_buffer = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        k_cache, v_cache, num_pages_per_seq = create_trtllm_paged_kv_cache(k, v, page_size)
    else:
        workspace_buffer, k_cache, v_cache, num_pages_per_seq = trtllm_state

    block_tables = torch.arange(
        batch_size * num_pages_per_seq, dtype=torch.int32, device=q.device
    ).reshape(batch_size, num_pages_per_seq)

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=q.device)

    kv_cache = (k_cache, v_cache)

    # q: [B*q_len, H_Q, D] for trtllm kernel
    q_input = q.reshape(-1, NUM_Q_HEADS, HEAD_DIM).contiguous()

    scale = 1.0 / (HEAD_DIM ** 0.5)

    kwargs = dict(
        query=q_input,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        window_left=-1,
        out_dtype=torch.bfloat16,
    )
    # q_len_per_req is needed for speculative (q_len > 1)
    if q_len > 1:
        kwargs["q_len_per_req"] = q_len

    o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(**kwargs)
    return o.reshape(batch_size, q_len, NUM_Q_HEADS, HEAD_DIM)


def create_trtllm_state(batch_size: int, seq_len: int, q_len: int,
                        kv_dtype: torch.dtype, page_size: int, device: str):
    """Pre-allocate TRTLLM workspace and KV cache for benchmarking."""
    workspace_buffer = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    if kv_dtype == torch.float8_e4m3fn:
        k = torch.rand(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM,
                       dtype=torch.float16, device=device).to(kv_dtype)
        v = torch.rand(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM,
                       dtype=torch.float16, device=device).to(kv_dtype)
    else:
        k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=kv_dtype, device=device)
        v = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=kv_dtype, device=device)
    k_cache, v_cache, num_pages_per_seq = create_trtllm_paged_kv_cache(k, v, page_size)
    return workspace_buffer, k_cache, v_cache, num_pages_per_seq


def calculate_theoretical_latency(batch_size: int, seq_len: int, q_len: int,
                                   q_dtype: torch.dtype, kv_dtype: torch.dtype):
    q_elem = dtype_element_size(q_dtype)
    kv_elem = dtype_element_size(kv_dtype)

    q_bytes = batch_size * q_len * NUM_Q_HEADS * HEAD_DIM * q_elem
    kv_bytes = 2 * batch_size * seq_len * NUM_KV_HEADS * HEAD_DIM * kv_elem
    o_bytes = batch_size * q_len * NUM_Q_HEADS * HEAD_DIM * q_elem
    total_bytes = q_bytes + kv_bytes + o_bytes

    traffic_gb = total_bytes / 1e9
    kv_gb = kv_bytes / 1e9
    theoretical_latency_us = traffic_gb / B200_PEAK_BW_TB_S / 1e3 * 1e6  # us

    # FLOPs: 2 ops × (QK^T + AV) × B × q_len × S × H_Q × D
    flops = 2 * 2 * batch_size * q_len * seq_len * NUM_Q_HEADS * HEAD_DIM

    return theoretical_latency_us, traffic_gb, kv_gb, flops


def bench_provider(provider_fn, q, k, v, warmup_iters=20, bench_iters=100, extra_arg=None):
    """Benchmark provider, returns latency in microseconds."""
    def call():
        if extra_arg is not None:
            return provider_fn(q, k, v, extra_arg)
        return provider_fn(q, k, v)

    for _ in range(warmup_iters):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        call()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / bench_iters * 1000  # us


def run_single_benchmark(
    batch_size: int,
    seq_len: int,
    q_len: int = 1,
    kv_dtype: torch.dtype = torch.bfloat16,
    page_size: int = 64,
    warmup_iters: int = 20,
    bench_iters: int = 100,
) -> List[BenchResult]:
    results = []

    major, minor = get_device_capability()
    q_dtype = torch.bfloat16

    q, k, v = create_decode_inputs(batch_size, seq_len, q_len, q_dtype, kv_dtype)
    theoretical_us, traffic_gb, kv_gb, flops = calculate_theoretical_latency(
        batch_size, seq_len, q_len, q_dtype, kv_dtype
    )
    kv_dtype_str = "fp8" if kv_dtype == torch.float8_e4m3fn else "bf16"
    print(f"  GPU: SM{major}{minor} | q_len={q_len} | kv_dtype={kv_dtype_str} | page_size={page_size}")
    print(f"  KV traffic: {kv_gb*1024:.1f} MB  |  Total: {traffic_gb*1024:.1f} MB  |  FLOPs: {flops/1e9:.2f} GFLOPs")

    results.append(BenchResult(
        batch_size=batch_size, seq_len=seq_len, provider="Theoretical",
        latency_us=theoretical_us, traffic_gb=traffic_gb,
        bandwidth_gb_s=traffic_gb / (theoretical_us * 1e-6),
    ))

    providers = {
        "PyTorch SDPA": (attn_pytorch_sdpa, None),
    }

    if check_fa4():
        providers["FA4"] = (attn_fa4_decode, None)

    if check_trtllm_gen():
        try:
            trtllm_state = create_trtllm_state(batch_size, seq_len, q_len, kv_dtype, page_size, 'cuda')
            providers["TRTLLM-GEN"] = (attn_trtllm_gen_decode, trtllm_state)
        except Exception as e:
            print(f"  TRTLLM-GEN state creation failed: {e}")

    for name, (fn, extra) in providers.items():
        try:
            latency_us = bench_provider(fn, q, k, v, warmup_iters, bench_iters, extra)
            results.append(BenchResult(
                batch_size=batch_size, seq_len=seq_len, provider=name,
                latency_us=latency_us, traffic_gb=traffic_gb,
                bandwidth_gb_s=traffic_gb / (latency_us * 1e-6),
            ))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    return results


def print_comparison_table(results: List[BenchResult]):
    print(f"\n{'Provider':>15} | {'Latency(us)':>12} | {'BW(GB/s)':>10} | {'vs Theoretical':>15}")
    print("-" * 65)
    theoretical = next(r for r in results if r.provider == "Theoretical")
    for r in sorted(results, key=lambda x: x.latency_us):
        if r.provider == "Theoretical":
            print(f"{r.provider:>15} | {r.latency_us:>12.2f} | {r.bandwidth_gb_s:>10.1f} | {'(peak, 100%)':>15}")
        else:
            ratio = r.latency_us / theoretical.latency_us
            eff = theoretical.latency_us / r.latency_us * 100
            print(f"{r.provider:>15} | {r.latency_us:>12.2f} | {r.bandwidth_gb_s:>10.1f} | {ratio:>6.2f}x ({eff:4.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax-M2.5 decode attention micro-benchmark"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--q-len", type=int, default=1,
                        help="Query tokens per request. 1=standard decode, 8=EAGLE speculative (production)")
    parser.add_argument("--kv-dtype", type=str, default="bf16", choices=["bf16", "fp8"],
                        help="KV cache dtype. fp8=FP8 E4M3 (production), bf16=BFloat16")
    parser.add_argument("--page-size", type=int, default=64,
                        help="KV cache page size. Production uses 64 (P64 in kernel name)")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "seq_sweep", "sweep"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ncu", action="store_true",
                        help="NCU mode: skip all other providers, run TRTLLM-GEN once for kernel profiling")
    args = parser.parse_args()

    kv_dtype = torch.float8_e4m3fn if args.kv_dtype == "fp8" else torch.bfloat16

    major, minor = get_device_capability()

    # NCU mode: run TRTLLM-GEN once for kernel profiling, skip everything else
    if args.ncu:
        import flashinfer.decode
        q_dtype = torch.bfloat16
        q, k, v = create_decode_inputs(args.batch_size, args.seq_len, args.q_len, q_dtype, kv_dtype)
        trtllm_state = create_trtllm_state(args.batch_size, args.seq_len, args.q_len, kv_dtype, args.page_size, "cuda")
        torch.cuda.synchronize()
        # single run - ncu will capture this
        attn_trtllm_gen_decode(q, k, v, trtllm_state)
        torch.cuda.synchronize()
        return

    print("=" * 80)
    print("MiniMax-M2.5 Decode Attention Benchmark with Baselines")
    print(f"Config: TP={TP_SIZE}, Per-GPU: {NUM_Q_HEADS} Q heads, {NUM_KV_HEADS} KV heads (GQA={GQA_RATIO})")
    print(f"Hardware: SM{major}{minor}  |  KV dtype: {args.kv_dtype}  |  q_len: {args.q_len}  |  page_size: {args.page_size}")
    print("=" * 80)

    print("\nAvailable providers:")
    print(f"  PyTorch SDPA: ✓ (always available, bf16 only)")
    print(f"  TRTLLM-GEN:   {'✓ (SM100 production kernel)' if check_trtllm_gen() else '✗'}")
    print(f"  FA4:          {'✓ (SM100 Blackwell)' if check_fa4() else '✗'}")

    if args.mode == "single":
        print(f"\nRunning: batch_size={args.batch_size}, seq_len={args.seq_len}")
        results = run_single_benchmark(
            args.batch_size, args.seq_len, args.q_len, kv_dtype, args.page_size
        )
        print_comparison_table(results)

        best = min((r for r in results if r.provider != "Theoretical"),
                   key=lambda x: x.latency_us, default=None)
        if best:
            theoretical = next(r for r in results if r.provider == "Theoretical")
            eff = theoretical.latency_us / best.latency_us * 100
            total_attn_ms = best.latency_us * NUM_LAYERS / 1000
            print(f"\nSummary: best={best.provider}  latency={best.latency_us:.1f}us  "
                  f"BW_eff={eff:.1f}%  total_attn={total_attn_ms:.2f}ms ({total_attn_ms/50*100:.1f}% of 50ms budget)")

    elif args.mode == "seq_sweep":
        seq_lens = [8192, 12288, 16384, 20480, 24576, 28672, 32768]
        print(f"\nSeq sweep (batch_size={args.batch_size}, q_len={args.q_len}, kv={args.kv_dtype}): {seq_lens}")
        print(f"{'seq_len':>10} | {'Provider':>14} | {'Latency(us)':>12} | {'BW(GB/s)':>10} | {'vs_Theo':>14} | {'KV(MB)':>8}")
        print("-" * 80)
        for sl in seq_lens:
            _, _, kv_gb, _ = calculate_theoretical_latency(
                args.batch_size, sl, args.q_len, torch.bfloat16, kv_dtype
            )
            results = run_single_benchmark(args.batch_size, sl, args.q_len, kv_dtype, args.page_size)
            theo = next(r for r in results if r.provider == "Theoretical")
            for r in results:
                if r.provider == "Theoretical":
                    ratio_str = "  (peak 100%)"
                else:
                    ratio = r.latency_us / theo.latency_us
                    eff = theo.latency_us / r.latency_us * 100
                    ratio_str = f"{ratio:.2f}x ({eff:4.1f}%)"
                print(f"{sl:>10} | {r.provider:>14} | {r.latency_us:>12.2f} | "
                      f"{r.bandwidth_gb_s:>10.1f} | {ratio_str:>14} | {kv_gb*1024:>8.1f}")

    else:  # sweep
        batch_sizes = [1, 8, 32, 64, 128] if args.quick else [1, 2, 4, 8, 16, 32, 64, 128]
        seq_lens = [1024, 4096, 8192, 32768] if args.quick else [1024, 2048, 4096, 8192, 16384, 32768]
        print(f"\nSweeping {len(batch_sizes)} batch sizes x {len(seq_lens)} seq lengths...")
        for bs in batch_sizes:
            for sl in seq_lens:
                print(f"\nbatch_size={bs}, seq_len={sl}:")
                try:
                    results = run_single_benchmark(bs, sl, args.q_len, kv_dtype, args.page_size)
                    print_comparison_table(results)
                except Exception as e:
                    print(f"  Error: {e}")


if __name__ == "__main__":
    main()
