# save this file as check_path.py
import sys
import os
import pprint
import traceback
import paddle.distributed as dist

def worker_fn(rank):
    print(f"\n--- [Rank {rank}] Starting Path and Environment Diagnostics ---")

    # 1. Print Current Working Directory
    try:
        cwd = os.getcwd()
        print(f"  [Rank {rank}] Current Working Directory (CWD): {cwd}")
    except Exception as e:
        print(f"  [Rank {rank}] ❌ FAILED to get CWD: {e}")

    # 2. Print sys.path
    print(f"  [Rank {rank}] Python Executable: {sys.executable}")
    print(f"  [Rank {rank}] sys.path:")
    pprint.pprint(sys.path)

    # 3. Check if project root is in sys.path
    #    Adjust this path if your project structure is different
    project_root_expected = "/home/aistudio/work/FastDeploy"
    if any(project_root_expected in p for p in sys.path):
        print(f"  [Rank {rank}] ✅ Project root '{project_root_expected}' seems to be in sys.path.")
    else:
        print(f"  [Rank {rank}] ❌ WARNING: Project root '{project_root_expected}' NOT FOUND in sys.path.")
        print(f"     This could be the reason for 'could not get source code' errors.")

    # 4. Attempt to import the problematic module chain directly
    problematic_imports = [
        "triton",
        "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
        "fastdeploy.model_executor.ops.triton_ops.minimax_mamba_kernels",
    ]
    
    all_imports_ok = True
    for module_name in problematic_imports:
        print(f"  [Rank {rank}] -> Attempting to import '{module_name}'...")
        try:
            __import__(module_name)
            print(f"  [Rank {rank}] ✅ Successfully imported '{module_name}'.")
        except Exception as e:
            all_imports_ok = False
            print(f"  [Rank {rank}] ❌ FAILED to import '{module_name}'.")
            print(f"     Error Type: {type(e).__name__}")
            print(f"     Error Msg: {e}")
            # If it's the OSError, it confirms our suspicion
            if isinstance(e, OSError) and 'could not get source code' in str(e):
                print("     >>> THIS IS THE CONFIRMED ROOT CAUSE! <<<")
            traceback.print_exc()

    print(f"--- [Rank {rank}] Diagnostics Finished ---")
    if not all_imports_ok:
        raise RuntimeError(f"Rank {rank} failed one or more import checks.")


def main():
    world_size = 8 # We are testing with 8 GPUs
    print("="*60)
    print("Starting Distributed Path Diagnostic Test...")
    print(f"Attempting to spawn {world_size} process(es).")
    print("="*60)
    
    try:
        dist.spawn(worker_fn, nprocs=world_size, join=True)
        print("\n" + "="*60)
        print("✅ ✅ ✅ All workers completed diagnostics without crashing!")
        print("Please review the path and import logs from each rank above.")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ❌ ❌ One or more workers crashed during diagnostics.")
        print(f"This strongly indicates a fundamental import problem. Review the traceback above.")
        print("="*60)

if __name__ == "__main__":
    main()