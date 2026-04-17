#!/usr/bin/env python3
"""Compare MoE internal values between FD and vLLM (rank 0).

Usage:
    python scripts/compare_moe_dump.py <dump_dir> [rank]
"""
import sys
import os
import numpy as np

dump_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dump_moe"
rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0

def load(prefix, suffix, layer=0):
    fd_path_new = f"{dump_dir}/fd_moe_r{rank}_l{layer}_{prefix}{suffix}.npy"
    fd_path_old = f"{dump_dir}/fd_moe_r{rank}_{prefix}{suffix}.npy"
    vllm_path = f"{dump_dir}/vllm_moe_r{rank}_l{layer}_{prefix}{suffix}.npy"
    fd = None
    if os.path.exists(fd_path_new):
        fd = np.load(fd_path_new)
    elif os.path.exists(fd_path_old):
        fd = np.load(fd_path_old)
    vllm = np.load(vllm_path) if os.path.exists(vllm_path) else None
    return fd, vllm

def compare(name, fd, vllm):
    if fd is None or vllm is None:
        print(f"  {name}: FD={'OK' if fd is not None else 'MISSING'} vLLM={'OK' if vllm is not None else 'MISSING'}")
        return
    fd_flat = fd.flatten().astype(np.float64)
    vllm_flat = vllm.flatten().astype(np.float64)
    if fd.shape != vllm.shape:
        print(f"  {name}: SHAPE MISMATCH! FD={fd.shape} vLLM={vllm.shape}")
        return
    dot = np.dot(fd_flat, vllm_flat)
    norm_fd = np.linalg.norm(fd_flat)
    norm_vllm = np.linalg.norm(vllm_flat)
    cosine = dot / (norm_fd * norm_vllm + 1e-12)
    max_diff = np.max(np.abs(fd_flat - vllm_flat))
    mean_diff = np.mean(np.abs(fd_flat - vllm_flat))
    status = "OK" if cosine > 0.9999 else ("SLIGHT" if cosine > 0.99 else "DIVERGE")
    print(f"  {name}: cosine={cosine:.6f}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  [{status}]")

print(f"Comparing MoE internals (rank {rank}):")
print(f"  dump_dir: {dump_dir}")
print()

# 1. Gate input (input to MoE block)
fd_gate, vllm_gate = load("", "gate_input")
compare("gate_input", fd_gate, vllm_gate)

# 2. topk_weights
fd_tw, vllm_tw = load("", "topk_weights")
compare("topk_weights", fd_tw, vllm_tw)

# 3. topk_ids (pre-align, before moe_align_block_size)
fd_ti, vllm_ti = load("", "topk_ids")
compare("topk_ids", fd_ti, vllm_ti)

# 4. sorted_token_ids
fd_st, vllm_st = load("", "sorted_token_ids")
compare("sorted_token_ids", fd_st, vllm_st)

# 5. expert_ids
fd_ei, vllm_ei = load("", "expert_ids")
compare("expert_ids", fd_ei, vllm_ei)

# 6. swiglu output
fd_sw, vllm_sw = load("", "swiglu")
compare("swiglu", fd_sw, vllm_sw)

print()
print("Divergence point analysis:")
print("  If gate_input diverges → problem is BEFORE MoE (attention/norm)")
print("  If topk_ids diverges → gate routing differs")
print("  If sorted_token_ids/expert_ids diverges → moe_align_block_size differs")
print("  If swiglu diverges → first Marlin GEMM (up_gate) differs")
print("  If all match but post_moe diverges → second Marlin GEMM (down) or output assembly differs")
