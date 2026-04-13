#!/usr/bin/env python3
"""
Simple test script to check FlashAttention backends availability.
"""

import torch

def check_flashinfer():
    """Check FlashInfer availability and API."""
    print("=" * 60)
    print("FlashInfer Check")
    print("=" * 60)
    
    try:
        import flashinfer
        print(f"✓ flashinfer imported, version: {getattr(flashinfer, '__version__', 'unknown')}")
        
        # List available functions
        funcs = [x for x in dir(flashinfer) if 'decode' in x.lower() or 'batch' in x.lower()]
        print(f"  Available decode/batch functions: {funcs[:10]}...")
        
        # Check specific APIs
        if hasattr(flashinfer, 'BatchDecodeWithPagedKVCacheWrapper'):
            print("  ✓ BatchDecodeWithPagedKVCacheWrapper available")
        else:
            print("  ✗ BatchDecodeWithPagedKVCacheWrapper NOT available")
            
        if hasattr(flashinfer, 'batch_decode_with_padded_kv_cache'):
            print("  ✓ batch_decode_with_padded_kv_cache available")
        else:
            print("  ✗ batch_decode_with_padded_kv_cache NOT available")
            
        return True
    except ImportError as e:
        print(f"✗ flashinfer import failed: {e}")
        return False


def check_fa3():
    """Check FlashAttention3 availability."""
    print("\n" + "=" * 60)
    print("FlashAttention3 Check")
    print("=" * 60)
    
    major, _ = torch.cuda.get_device_capability()
    print(f"Device capability: SM{major}x")
    
    if major != 9:
        print(f"✗ FA3 requires SM90 (Hopper), current is SM{major}")
        return False
    
    try:
        from flash_attn import flash_attn_func
        print("✓ flash_attn imported from flash_attn")
        return True
    except ImportError:
        pass
    
    try:
        from flash_attn_interface import flash_attn_func
        print("✓ flash_attn_func imported from flash_attn_interface")
        return True
    except ImportError:
        pass
    
    print("✗ FA3 not available (neither flash_attn nor flash_attn_interface)")
    return False


def check_fa4():
    """Check FlashAttention4 availability."""
    print("\n" + "=" * 60)
    print("FlashAttention4 Check")
    print("=" * 60)
    
    major, _ = torch.cuda.get_device_capability()
    print(f"Device capability: SM{major}x")
    
    if major != 10:
        print(f"✗ FA4 requires SM100 (Blackwell), current is SM{major}")
        return False
    
    # Try multiple import paths
    import_paths = [
        ("flash_attn_4", "flash_attn_func"),
        ("flash_attn", "flash_attn_func"),
        ("flash_attn_interface", "flash_attn_func"),
    ]
    
    for module, func in import_paths:
        try:
            mod = __import__(module)
            if hasattr(mod, func):
                print(f"✓ {func} available in {module}")
                return True
            # Check submodule
            for sub in dir(mod):
                submod = getattr(mod, sub)
                if hasattr(submod, func):
                    print(f"✓ {func} available in {module}.{sub}")
                    return True
        except ImportError:
            continue
    
    print("✗ FA4 not found in any known module path")
    print("  Tried: flash_attn_4, flash_attn, flash_attn_interface")
    
    # Additional check - list installed packages
    print("\n  Checking installed packages with 'flash' in name:")
    import subprocess
    try:
        result = subprocess.run(
            ["pip", "list"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        for line in result.stdout.split('\n'):
            if 'flash' in line.lower() or 'attn' in line.lower():
                print(f"    {line}")
    except Exception as e:
        print(f"  Could not run pip list: {e}")
    
    return False


def test_fa4_simple():
    """Simple FA4 test."""
    print("\n" + "=" * 60)
    print("FA4 Simple Test")
    print("=" * 60)
    
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        print("Skipping FA4 test - not on SM100")
        return
    
    try:
        from flash_attn_4 import flash_attn_func
        print("✓ Import successful")
        
        # Create simple inputs
        batch, seq, heads, dim = 2, 1024, 32, 128
        q = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16, device='cuda')
        
        # Run
        out, lse = flash_attn_func(q, k, v, causal=True)
        print(f"✓ Execution successful, output shape: {out.shape}")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
    except Exception as e:
        print(f"✗ Execution failed: {e}")


if __name__ == "__main__":
    check_flashinfer()
    check_fa3()
    check_fa4()
    test_fa4_simple()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: SM{major}{minor}")
    print(f"Recommended backend: ", end="")
    if major == 10:
        print("FA4 (Blackwell)")
    elif major == 9:
        print("FA3 (Hopper)")
    else:
        print("FlashInfer (Ampere+)")
