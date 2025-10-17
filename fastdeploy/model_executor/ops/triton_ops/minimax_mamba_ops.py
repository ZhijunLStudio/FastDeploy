# work/FastDeploy/fastdeploy/model_executor/ops/triton_ops/minimax_mamba_ops.py

from typing import Optional
import paddle
import paddle.nn.functional as F
from paddleformers.utils.log import logger
import triton
import pprint

# 导入你已经准备好的底层 Triton JIT Kernels
from .minimax_mamba_kernels import (
    _fwd_diag_kernel,
    _fwd_kv_parallel,
    _fwd_kv_reduce,
    _fwd_none_diag_kernel,
    _linear_attn_decode_kernel,
)

# 保留打印函数，用于调试
def print_tensor_stats(tensor, name):
    """打印Paddle张量的统计信息 (强制 float32)"""
    if tensor is None:
        logger.info(f"[FD DEBUG_KERNEL] {name} is None")
        return
    with paddle.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if tensor.numel() > 0:
            tensor_float = tensor.astype('float32')
            tensor_cpu = tensor_float.cpu()
            stats["max"] = f"{tensor_cpu.max().item():.6f}"
            stats["min"] = f"{tensor_cpu.min().item():.6f}"
            stats["mean"] = f"{tensor_cpu.mean().item():.6f}"
            stats["std"] = f"{tensor_cpu.std().item():.6f}"
            flat_data = tensor_cpu.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        stats_str = f"\n--- [FD DEBUG_KERNEL] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n"
        logger.info(stats_str)

# 这是一个基于 paddle.autograd.Function 的包装类，用于调用 Triton kernels
# 它的作用类似于 PyTorch 中的 torch.autograd.Function
class _Attention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history_in):
        # 确保输入张量在内存中是连续的
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # ==================== 核心修复：在 PyLayer 内部创建副本 ====================
        # 创建一个与输入 kv_history 形状和类型相同的、用于计算的张量。
        # 这是为了避免原地修改传入的叶子节点张量。
        kv_history_compute = paddle.clone(kv_history_in)
        # =====================================================================

        # 获取输入维度
        b, h, n, d = q.shape
        e = v.shape[-1]

        # 初始化输出张量
        o = paddle.empty(shape=[b, h, n, e], dtype=q.dtype)
        
        # --- [后续所有 Triton Kernel 调用逻辑保持不变] ---
        # ... (设置 BLOCK, CBLOCK, 计算 k_decay 等) ...

        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        CBLOCK_DIAG = 32
        NUM_CBLOCK_DIAG = BLOCK // CBLOCK_DIAG
        assert BLOCK % CBLOCK_DIAG == 0
        array = paddle.arange(0, BLOCK) + 1
        array_float = array.astype("float32")
        k_decay = paddle.exp(-s * (BLOCK - array_float.reshape([1, -1])))

        # Step 1
        grid_diag = (b * h * NUM_BLOCK, NUM_CBLOCK_DIAG)
        _fwd_diag_kernel[grid_diag](
            q, k, v, o, s,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, CBLOCK=CBLOCK_DIAG,
        )

        # Step 2
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK
        CBLOCK_KV_AND_NON_DIAG = 64
        NUM_CBLOCK_KV_AND_NON_DIAG = BLOCK // CBLOCK_KV_AND_NON_DIAG
        kv = paddle.empty(shape=[b, h, NUM_BLOCK, d, e], dtype="float32")
        grid_kv_parallel = (b * h, NUM_BLOCK)
        _fwd_kv_parallel[grid_kv_parallel](
            k, v, k_decay, kv,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK, NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK_KV_AND_NON_DIAG, NUM_CBLOCK=NUM_CBLOCK_KV_AND_NON_DIAG,
        )

        # Step 3: 将新创建的 kv_history_compute 传入，让它被原地修改
        grid_kv_reduce = (b * h, NUM_FBLOCK)
        _fwd_kv_reduce[grid_kv_reduce](
            s, kv, kv_history_compute,  # <--- 使用副本
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK,
        )

        # Step 4
        grid_none_diag = (b * h, NUM_BLOCK * NUM_CBLOCK_KV_AND_NON_DIAG)
        _fwd_none_diag_kernel[grid_none_diag](
            q, o, s, kv,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, E_FBLOCK=E_FBLOCK,
            CBLOCK=CBLOCK_KV_AND_NON_DIAG, NUM_CBLOCK=NUM_CBLOCK_KV_AND_NON_DIAG,
        )
        
        # 返回计算结果和被更新后的 kv_history 副本
        return o, kv_history_compute

    @staticmethod
    def backward(ctx, grad_output, grad_kv_history):
        raise NotImplementedError("Backward pass for lightning_attention is not implemented")

def lightning_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    slope_rate: paddle.Tensor,
    kv_history: Optional[paddle.Tensor] = None,
    is_profiling: bool = False,
    block_size: int = 256,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    
    if is_profiling:
        logger.warning("<<<<< RUNNING in PROFILING MODE for LIGHTNING ATTENTION! >>>>>")
        logger.warning("<<<<< Bypassing actual computation and returning dummy tensors. >>>>>")
        dummy_output = paddle.zeros_like(v)
        dummy_kv_state = paddle.zeros_like(kv_history) if kv_history is not None else paddle.zeros(shape=[q.shape[0], q.shape[1], q.shape[3], v.shape[3]], dtype=v.dtype)
        return dummy_output, dummy_kv_state

    logger.info("<<<<< RUNNING TRITON KERNEL FOR LIGHTNING ATTENTION! >>>>>")
    
    original_dtype = q.dtype
    
    if slope_rate.dim() == 1:
        slope_rate = slope_rate.reshape([1, -1, 1, 1])

    if kv_history is None:
        kv_history = paddle.zeros(shape=[q.shape[0], q.shape[1], q.shape[3], v.shape[3]], dtype="float32")
    
    # ==================== 核心修改：移除这里的 clone ====================
    # kv_history = kv_history.clone().contiguous() # <--- 移除这一行
    # 直接将原始的 kv_history 传给 apply
    # =================================================================

    output, updated_kv_history = _Attention.apply(q, k, v, slope_rate, kv_history)

    return output.astype(original_dtype), updated_kv_history.astype(original_dtype)



# ----------------------------------------------------------------------------------
#  decode 函数保持不变，因为它已经使用了 Triton Kernel
# ----------------------------------------------------------------------------------
def linear_decode_forward_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    kv_caches: paddle.Tensor,
    slope_rate: paddle.Tensor,
    slot_idx: paddle.Tensor,
    BLOCK_SIZE: int = 32,
) -> paddle.Tensor:
    B, H, _, D = q.shape
    assert tuple(k.shape) == (B, H, 1, D), f"Shape of k is {k.shape}, expected {(B, H, 1, D)}"
    assert tuple(v.shape) == (B, H, 1, D), f"Shape of v is {v.shape}, expected {(B, H, 1, D)}"
    from einops import rearrange
    output = paddle.empty_like(q)
    grid = (B, H, triton.cdiv(D, BLOCK_SIZE))

    # ==================== 核心修复 ====================
    qkv_b_stride, qkv_h_stride = q.strides[0], q.strides[1]
    cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride = (kv_caches.strides[0], kv_caches.strides[1], kv_caches.strides[2], kv_caches.strides[3])
    # ================================================

    _linear_attn_decode_kernel[grid](q, k, v, kv_caches, slope_rate, slot_idx, output, D=D, qkv_b_stride=qkv_b_stride, qkv_h_stride=qkv_h_stride, cache_b_stride=cache_b_stride, cache_h_stride=cache_h_stride, cache_d0_stride=cache_d0_stride, cache_d1_stride=cache_d1_stride, BLOCK_SIZE=BLOCK_SIZE)
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()