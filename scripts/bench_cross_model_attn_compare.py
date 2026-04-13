#!/usr/bin/env python3
"""
Cross-model decode attention kernel comparison benchmark.

Compare MiniMax-M2.5 against other popular MoE models:
- MiniMax-M2.5: GQA 48:8, head_dim=128
- DeepSeek-V3: MLA (Multi-head Latent Attention)
- Qwen3-235B: GQA 32:8, head_dim=128
- DeepSeek-R1: MLA
- Mixtral 8x22B: GQA 48:8, head_dim=128

This helps understand if MiniMax-M2.5's 40% attention overhead is abnormal.

Note on attention backends:
- FA3: SM90 (Hopper H100/H200) only
- FA4: SM100 (Blackwell B200/B300/GB200) only
- FlashInfer: SM80+ (all modern NVIDIA GPUs)

Usage:
    python scripts/bench_cross_model_attn_compare.py --batch-size 64 --seq-len 8192
    python scripts/bench_cross_model_attn_compare.py --mode sweep --quick
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model attention configuration."""
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    tp_size: int
    attention_type: str  # "GQA" or "MLA"
    
    @property
    def per_gpu_q_heads(self) -> int:
        return self.num_q_heads // self.tp_size
    
    @property
    def per_gpu_kv_heads(self) -> int:
        return self.num_kv_heads // self.tp_size
    
    @property
    def gqa_ratio(self) -> int:
        return self.per_gpu_q_heads // max(1, self.per_gpu_kv_heads)


# Popular MoE model configurations
MODEL_CONFIGS = {
    "MiniMax-M2.5": ModelConfig(
        name="MiniMax-M2.5",
        num_q_heads=48,
        num_kv_heads=8,
        head_dim=128,
        num_layers=62,
        tp_size=4,
        attention_type="GQA",
    ),
    "DeepSeek-V3": ModelConfig(
        name="DeepSeek-V3",
        num_q_heads=128,  # MLA: query heads
        num_kv_heads=128,  # MLA: compressed latent
        head_dim=128,  # latent dim for V3 MLA
        num_layers=61,
        tp_size=8,
        attention_type="MLA",
    ),
    "Qwen3-235B": ModelConfig(
        name="Qwen3-235B",
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_layers=60,
        tp_size=4,
        attention_type="GQA",
    ),
    "DeepSeek-R1": ModelConfig(
        name="DeepSeek-R1",
        num_q_heads=128,
        num_kv_heads=128,
        head_dim=128,
        num_layers=61,
        tp_size=8,
        attention_type="MLA",
    ),
    "Mixtral-8x22B": ModelConfig(
        name="Mixtral-8x22B",
        num_q_heads=48,
        num_kv_heads=8,
        head_dim=128,
        num_layers=56,
        tp_size=4,
        attention_type="GQA",
    ),
    "LLaMA-3.1-70B": ModelConfig(
        name="LLaMA-3.1-70B",
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_layers=80,
        tp_size=4,
        attention_type="GQA",
    ),
}


@dataclass
class BenchmarkResult:
    model: str
    batch_size: int
    seq_len: int
    latency_us: float
    traffic_bytes: int
    bandwidth_gb_s: float
    total_attn_ms: float


def calculate_memory_traffic(config: ModelConfig, batch_size: int, seq_len: int) -> int:
    """Calculate memory traffic for decode attention."""
    element_size = 2  # bf16
    
    per_gpu_q = config.per_gpu_q_heads
    per_gpu_kv = config.per_gpu_kv_heads
    head_dim = config.head_dim
    
    if config.attention_type == "GQA":
        # Q read: batch_size * per_gpu_q * head_dim
        # K read: batch_size * seq_len * per_gpu_kv * head_dim
        # V read: batch_size * seq_len * per_gpu_kv * head_dim
        # O write: batch_size * per_gpu_q * head_dim
        q_bytes = batch_size * per_gpu_q * head_dim * element_size
        kv_bytes = 2 * batch_size * seq_len * per_gpu_kv * head_dim * element_size
        o_bytes = batch_size * per_gpu_q * head_dim * element_size
        return q_bytes + kv_bytes + o_bytes
    
    elif config.attention_type == "MLA":
        # MLA has different memory pattern:
        # Q read: batch_size * per_gpu_q * head_dim
        # KV latent read: batch_size * seq_len * head_dim (compressed)
        # Plus rope component if separate
        # For simplicity, assume latent is similar to 1 KV head per batch position
        q_bytes = batch_size * per_gpu_q * head_dim * element_size
        kv_bytes = 2 * batch_size * seq_len * head_dim * element_size  # compressed latent
        o_bytes = batch_size * per_gpu_q * head_dim * element_size
        return q_bytes + kv_bytes + o_bytes
    
    return 0


def simulate_decode_attention(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> float:
    """
    Simulate decode attention for a model config.
    Returns latency in microseconds.
    """
    per_gpu_q = config.per_gpu_q_heads
    per_gpu_kv = config.per_gpu_kv_heads
    head_dim = config.head_dim
    
    # Create inputs
    q = torch.randn(batch_size, per_gpu_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, per_gpu_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, per_gpu_kv, head_dim, dtype=dtype, device=device)
    
    gqa_ratio = per_gpu_q // max(1, per_gpu_kv)
    
    # Warmup
    for _ in range(10):
        if config.attention_type == "GQA" and gqa_ratio > 1:
            # Expand KV for GQA
            k_exp = k.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1).reshape(
                batch_size, seq_len, per_gpu_q, head_dim
            )
            v_exp = v.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1).reshape(
                batch_size, seq_len, per_gpu_q, head_dim
            )
        else:
            k_exp, v_exp = k, v
        
        q_t = q.unsqueeze(2)
        k_t = k_exp.transpose(1, 2)
        v_t = v_exp.transpose(1, 2)
        
        _ = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=True,
            scale=1.0 / (head_dim ** 0.5),
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(50):
        if config.attention_type == "GQA" and gqa_ratio > 1:
            k_exp = k.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1).reshape(
                batch_size, seq_len, per_gpu_q, head_dim
            )
            v_exp = v.unsqueeze(3).expand(-1, -1, -1, gqa_ratio, -1).reshape(
                batch_size, seq_len, per_gpu_q, head_dim
            )
        else:
            k_exp, v_exp = k, v
        
        q_t = q.unsqueeze(2)
        k_t = k_exp.transpose(1, 2)
        v_t = v_exp.transpose(1, 2)
        
        _ = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=True,
            scale=1.0 / (head_dim ** 0.5),
        )
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / 50
    return latency_ms * 1000  # Convert to microseconds


def run_model_comparison(
    batch_size: int,
    seq_len: int,
    models: List[str] = None,
) -> List[BenchmarkResult]:
    """Run comparison across models."""
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    
    results = []
    
    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            continue
        
        config = MODEL_CONFIGS[model_name]
        
        try:
            latency_us = simulate_decode_attention(config, batch_size, seq_len)
            traffic_bytes = calculate_memory_traffic(config, batch_size, seq_len)
            bandwidth_gb_s = (traffic_bytes / 1e9) / (latency_us * 1e-6)
            total_attn_ms = latency_us * config.num_layers / 1000
            
            results.append(BenchmarkResult(
                model=model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                latency_us=latency_us,
                traffic_bytes=traffic_bytes,
                bandwidth_gb_s=bandwidth_gb_s,
                total_attn_ms=total_attn_ms,
            ))
        except Exception as e:
            print(f"Warning: {model_name} benchmark failed: {e}")
    
    return results


def print_comparison(results: List[BenchmarkResult], target_decode_ms: float = 50.0):
    """Print comparison table."""
    print(f"\n{'Model':>18} | {'Attn Type':>10} | {'Per-Layer (us)':>14} | {'Total Attn (ms)':>15} | {'Share @ {target_decode_ms}ms':>18}")
    print("-" * 90)
    
    for r in sorted(results, key=lambda x: x.total_attn_ms):
        config = MODEL_CONFIGS[r.model]
        share = (r.total_attn_ms / target_decode_ms) * 100
        status = "⚠️ HIGH" if share > 40 else "✓ OK"
        
        print(
            f"{r.model:>18} | {config.attention_type:>10} | {r.latency_us:>14.2f} | "
            f"{r.total_attn_ms:>15.2f} | {share:>17.1f}% {status}"
        )


def print_model_configs():
    """Print model configurations."""
    print("\nModel Configurations:")
    print("=" * 100)
    print(f"{'Model':>18} | {'Type':>6} | {'Q Heads':>8} | {'KV Heads':>8} | {'Head Dim':>9} | {'Layers':>7} | {'TP':>3} | {'GQA Ratio':>10}")
    print("-" * 100)
    
    for name, config in MODEL_CONFIGS.items():
        print(
            f"{name:>18} | {config.attention_type:>6} | {config.per_gpu_q_heads:>8} | "
            f"{config.per_gpu_kv_heads:>8} | {config.head_dim:>9} | {config.num_layers:>7} | "
            f"{config.tp_size:>3} | {config.gqa_ratio:>10}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model decode attention comparison"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sweep"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--target-decode-ms", type=float, default=50.0,
                        help="Target decode latency in ms for share calculation")
    args = parser.parse_args()
    
    print("=" * 100)
    print("Cross-Model Decode Attention Kernel Comparison")
    print("=" * 100)
    
    print_model_configs()
    
    print(f"\nBenchmark config: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"Target decode latency: {args.target_decode_ms} ms/token")
    
    if args.mode == "single":
        results = run_model_comparison(args.batch_size, args.seq_len)
        print_comparison(results, args.target_decode_ms)
        
        # Analysis
        print("\n" + "=" * 100)
        print("Analysis:")
        minimax_result = next((r for r in results if r.model == "MiniMax-M2.5"), None)
        
        if minimax_result:
            print(f"\nMiniMax-M2.5 attention overhead: {minimax_result.total_attn_ms:.2f} ms")
            share = minimax_result.total_attn_ms / args.target_decode_ms * 100
            print(f"Share of {args.target_decode_ms}ms budget: {share:.1f}%")
            
            if share > 40:
                print("\n⚠️  MiniMax-M2.5 attention share is HIGH (>40%)")
                print("\nPossible reasons:")
                print("  1. Long context (seq_len > 8K) increases O(n) cost")
                print("  2. EP=4 reduces MoE time, making attention appear larger")
                print("  3. Small batch size leads to poor kernel utilization")
                
                # Compare with other models
                other_results = [r for r in results if r.model != "MiniMax-M2.5"]
                if other_results:
                    avg_share = sum(
                        r.total_attn_ms / args.target_decode_ms * 100 
                        for r in other_results
                    ) / len(other_results)
                    print(f"\n  Average attention share for other models: {avg_share:.1f}%")
                    print(f"  MiniMax-M2.5 is {share/avg_share:.1f}x the average")
            else:
                print(f"\n✓ MiniMax-M2.5 attention share ({share:.1f}%) is within normal range")
    
    else:
        # Sweep mode
        if args.quick:
            batch_sizes = [1, 8, 32, 64, 128]
            seq_lens = [1024, 4096, 8192, 32768]
        else:
            batch_sizes = [1, 8, 16, 32, 64, 128]
            seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
        
        for bs in batch_sizes:
            for sl in seq_lens:
                print(f"\n{'='*80}")
                print(f"batch_size={bs}, seq_len={sl}")
                print("=" * 80)
                
                results = run_model_comparison(bs, sl)
                print_comparison(results, args.target_decode_ms)


if __name__ == "__main__":
    main()
