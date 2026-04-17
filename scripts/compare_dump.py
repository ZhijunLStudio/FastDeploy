"""
Compare FD vs vLLM dumped hidden_states layer by layer (per-rank).

Usage:
    python scripts/compare_dump.py [dump_dir] [rank]

Default dump_dir = /tmp/dump_compare, rank = 0
Expects files like:
    fd/fd_r0_l0_post_norm1.npy
    vllm/vllm_r0_l0_post_norm1.npy
"""
import sys, os, glob
import numpy as np

DUMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dump_compare"
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
FD_DIR = os.path.join(DUMP_DIR, "fd")
VLLM_DIR = os.path.join(DUMP_DIR, "vllm")


def stats(arr):
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "shape": arr.shape,
    }


def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def max_abs_diff(a, b):
    return float(np.abs(a.flatten() - b.flatten()).max())


def mean_abs_diff(a, b):
    return float(np.abs(a.flatten() - b.flatten()).mean())


def relative_err(a, b):
    """Mean relative error: |a-b| / (|b| + 1e-8)"""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    return float(np.mean(np.abs(a_flat - b_flat) / (np.abs(b_flat) + 1e-8)))


def compare_files(fd_path, vllm_path, label):
    fd_arr = np.load(fd_path)
    vllm_arr = np.load(vllm_path)

    fd_s = stats(fd_arr)
    vl_s = stats(vllm_arr)
    shape_match = fd_arr.shape == vllm_arr.shape

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  FD   shape={fd_s['shape']}, mean={fd_s['mean']:.6f}, std={fd_s['std']:.6f}, "
          f"max={fd_s['max']:.6f}, min={fd_s['min']:.6f}")
    print(f"  vLLM shape={vl_s['shape']}, mean={vl_s['mean']:.6f}, std={vl_s['std']:.6f}, "
          f"max={vl_s['max']:.6f}, min={vl_s['min']:.6f}")

    if shape_match:
        cos = cosine_sim(fd_arr, vllm_arr)
        mad = max_abs_diff(fd_arr, vllm_arr)
        mead = mean_abs_diff(fd_arr, vllm_arr)
        rel = relative_err(fd_arr, vllm_arr)
        print(f"  cosine_sim={cos:.8f}, max_abs_diff={mad:.8f}")
        print(f"  mean_abs_diff={mead:.8f}, rel_err={rel:.8f}")
    else:
        cos, mad = -1.0, -1.0
        mead = -1.0
        rel = -1.0
        print(f"  [shape mismatch - stats comparison only]")

    return cos, mad, mead, rel


def main():
    prefix_fd = f"fd_r{RANK}_"
    prefix_vl = f"vllm_r{RANK}_"

    fd_files = glob.glob(os.path.join(FD_DIR, f"{prefix_fd}*.npy"))
    vllm_files = glob.glob(os.path.join(VLLM_DIR, f"{prefix_vl}*.npy"))

    if not fd_files:
        print(f"No FD dump files found in {FD_DIR} for rank {RANK}")
        return
    if not vllm_files:
        print(f"No vLLM dump files found in {VLLM_DIR} for rank {RANK}")
        return

    # Build key mapping: strip rank prefix
    fd_keys = {}
    for f in fd_files:
        basename = os.path.basename(f)
        key = basename.replace(prefix_fd, "").replace(".npy", "")
        fd_keys[key] = f

    vllm_keys = {}
    for f in vllm_files:
        basename = os.path.basename(f)
        key = basename.replace(prefix_vl, "").replace(".npy", "")
        vllm_keys[key] = f

    # Sort keys: embed first, then l0, l1, ..., final_norm last
    def sort_key(k):
        if k == "embed":
            return (-1, "")
        if k == "final_norm":
            return (999, "")
        # l0_post_norm1 -> (0, "post_norm1")
        parts = k.split("_", 1)
        layer_num = int(parts[0][1:]) if parts[0].startswith("l") else 0
        sublayer = parts[1] if len(parts) > 1 else ""
        return (layer_num, sublayer)

    all_keys = sorted(set(fd_keys.keys()) & set(vllm_keys.keys()), key=sort_key)

    results = []
    for key in all_keys:
        cos, mad, mead, rel = compare_files(fd_keys[key], vllm_keys[key], f"rank{RANK} / {key}")
        results.append((key, cos, mad, mead, rel))

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY (rank={RANK})")
    print(f"{'='*80}")
    print(f"  {'Layer':<25} {'Cosine':>10} {'MaxAbs':>12} {'MeanAbs':>12} {'RelErr':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for key, cos, mad, mead, rel in results:
        flag = ""
        if cos >= 0:
            if cos < 0.99:
                flag = " *** DIVERGE"
            elif cos < 0.9999:
                flag = " * slight"
        print(f"  {key:<25} {cos:>10.6f} {mad:>12.8f} {mead:>12.8f} {rel:>10.6f}{flag}")


if __name__ == "__main__":
    main()
