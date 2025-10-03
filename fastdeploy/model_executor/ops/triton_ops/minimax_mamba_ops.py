# fastdeploy/model_executor/ops/triton_ops/minimax_mamba_ops.py

from typing import Optional
import paddle
import triton
from einops import rearrange
from .minimax_mamba_kernels import (_fwd_diag_kernel, _fwd_kv_parallel,
                                    _fwd_kv_reduce, _fwd_none_diag_kernel,
                                    _linear_attn_decode_kernel)

def lightning_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    slope_rate: paddle.Tensor,
    kv_history: Optional[paddle.Tensor] = None
) -> tuple[paddle.Tensor, paddle.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if slope_rate.dim() > 1:
        slope_rate = slope_rate.squeeze()
    slope_rate = slope_rate.contiguous()
    B, H, N, D = q.shape
    E = v.shape[-1]
    compute_dtype = q.dtype
    o = paddle.empty_like(v)
    if kv_history is None:
        kv_history_in = paddle.zeros((B, H, D, E), dtype=paddle.float32).to(q.place)
    else:
        kv_history_in = kv_history.clone().contiguous()
    BLOCK = 256
    NUM_BLOCK = triton.cdiv(N, BLOCK)
    CBLOCK_diag = 32
    NUM_CBLOCK_diag = BLOCK // CBLOCK_diag
    grid_diag = (B * H * NUM_BLOCK, NUM_CBLOCK_diag)
    _fwd_diag_kernel[grid_diag](q, k, v, o, slope_rate, b=B, h=H, n=N, d=D, e=E, BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, CBLOCK=CBLOCK_diag)
    array = (paddle.arange(0, BLOCK, dtype='float32') + 1).to(q.place)
    k_decay = paddle.exp(-slope_rate.reshape((H, 1)) * (BLOCK - array.reshape((1, -1))))
    k_decay = k_decay.astype(compute_dtype)
    NUM_FBLOCK = 1
    D_FBLOCK = D // NUM_FBLOCK
    E_FBLOCK = E // NUM_FBLOCK
    CBLOCK_kv = 64
    NUM_CBLOCK_kv = BLOCK // CBLOCK_kv
    kv_intermediate = paddle.empty((B, H, NUM_BLOCK, D, E), dtype=paddle.float32).to(q.place)
    grid_kv_parallel = (B * H, NUM_BLOCK)
    _fwd_kv_parallel[grid_kv_parallel](k, v, k_decay, kv_intermediate, b=B, h=H, n=N, d=D, e=E, BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK, NUM_FBLOCK=NUM_FBLOCK, CBLOCK=CBLOCK_kv, NUM_CBLOCK=NUM_CBLOCK_kv)
    grid_kv_reduce = (B * H, NUM_FBLOCK)
    _fwd_kv_reduce[grid_kv_reduce](slope_rate, kv_intermediate, kv_history_in, b=B, h=H, n=N, d=D, e=E, BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK)
    grid_none_diag = (B * H, NUM_BLOCK * NUM_CBLOCK_diag, NUM_FBLOCK)
    _fwd_none_diag_kernel[grid_none_diag](q, o, slope_rate, kv_intermediate, b=B, h=H, n=N, d=D, e=E, BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, E_FBLOCK=E_FBLOCK, CBLOCK=CBLOCK_diag, NUM_CBLOCK=NUM_CBLOCK_diag)
    return o, kv_history_in

def linear_decode_forward_triton(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, kv_caches: paddle.Tensor, slope_rate: paddle.Tensor, slot_idx: paddle.Tensor, BLOCK_SIZE: int = 32) -> paddle.Tensor:
    B, H, _, D = q.shape
    assert tuple(k.shape) == (B, H, 1, D), f"Shape of k is {k.shape}, expected {(B, H, 1, D)}"
    assert tuple(v.shape) == (B, H, 1, D), f"Shape of v is {v.shape}, expected {(B, H, 1, D)}"
    output = paddle.empty_like(q)
    grid = (B, H, triton.cdiv(D, BLOCK_SIZE))
    qkv_b_stride, qkv_h_stride = q.strides[0], q.strides[1]
    cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride = kv_caches.strides[0], kv_caches.strides[1], kv_caches.strides[2], kv_caches.strides[3]
    _linear_attn_decode_kernel[grid](q, k, v, kv_caches, slope_rate, slot_idx, output, D=D, qkv_b_stride=qkv_b_stride, qkv_h_stride=qkv_h_stride, cache_b_stride=cache_b_stride, cache_h_stride=cache_h_stride, cache_d0_stride=cache_d0_stride, cache_d1_stride=cache_d1_stride, BLOCK_SIZE=BLOCK_SIZE)
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()