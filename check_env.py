# save this file as check_spawn_env.py
import paddle.distributed as dist
import os
import sys
import traceback

def worker_fn(rank):
    print(f"\n--- [Rank {rank}] Worker started. ---")
    
    # 1. Check environment inside the spawned process
    print(f"  [Rank {rank}] Python executable: {sys.executable}")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"  [Rank {rank}] Inherited CUDA_VISIBLE_DEVICES: {cuda_visible}")

    try:
        gpu_count = paddle.device.cuda.device_count()
        print(f"  [Rank {rank}] paddle.device.cuda.device_count(): {gpu_count}")
        if gpu_count == 0:
            print(f"  [Rank {rank}] ❌ CRITICAL: No GPUs visible inside spawned process!")
            raise RuntimeError("Spawned process cannot see GPUs.")
    except Exception as e:
        print(f"  [Rank {rank}] ❌ CRITICAL: Error checking GPUs inside spawned process.")
        traceback.print_exc()
        raise e

    # 2. Set device for this worker
    paddle.set_device(f'gpu:{rank}')
    print(f"  [Rank {rank}] Successfully set device to gpu:{rank}")

    # 3. Test Triton
    print(f"  [Rank {rank}] -> Attempting to import and use Triton...")
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        size = 128
        x = paddle.rand((size,)).cuda()
        y = paddle.rand((size,)).cuda()
        output = paddle.empty((size,)).cuda()
        
        grid = (triton.cdiv(size, 128),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=128)
        
        assert paddle.allclose(output, x + y)
        print(f"  [Rank {rank}] ✅ Triton JIT test PASSED.")
    except Exception as e:
        print(f"  [Rank {rank}] ❌ Triton JIT test FAILED.")
        traceback.print_exc()
        raise e

    print(f"--- [Rank {rank}] Worker finished successfully. ---")


def main():
    try:
        world_size = paddle.device.cuda.device_count()
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
            world_size = min(world_size, len(visible_devices))
            print(f"Main process sees CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print("CUDA_VISIBLE_DEVICES is not set. Paddle will see all available GPUs.")
    except Exception:
        world_size = 0

    if world_size == 0:
        print("Main process can't see any CUDA GPUs. Aborting.")
        return
        
    print("="*60)
    print("Starting Spawn Environment Diagnostic Test...")
    print(f"Attempting to spawn {world_size} process(es).")
    print("="*60)
    
    try:
        # We pass no args, spawn should only pass rank.
        dist.spawn(worker_fn, nprocs=world_size, join=True)
        print("\n" + "="*60)
        print("✅ ✅ ✅ All workers completed without crashing!")
        print("Please check logs above for individual test results (✅ or ❌).")
        print("="*60)
    except Exception:
        print("\n" + "="*60)
        print("❌ ❌ ❌ dist.spawn reported a failure in at least one worker.")
        print("Review the traceback above from the failed worker.")
        print("="*60)

if __name__ == "__main__":
    main()