# mamba_validation/generate_golden_data.py
import torch
import numpy as np
import os
from vllm.model_executor.layers.lightning_attn import lightning_attention, linear_decode_forward_triton

def generate_prefill_data():
    print("--- Generating Prefill Golden Data (using float32) ---")
    B, H, N, D = 1, 64, 256, 128
    torch.manual_seed(0)
    
    # 使用 float32 生成，以获得最精确的基准
    q = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    slope_rate = (torch.randn(H, 1, 1, dtype=torch.float32, device="cuda") * 0.1).abs()
    kv_history_in = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") # 使用 randn 初始化
    
    # 调用 vLLM 原生函数
    output, kv_out_combined = lightning_attention(q, k, v, slope_rate, kv_history=kv_history_in.clone())
    
    # ++++++++++++++++ 关键修正：正确提取最终的 kv_history ++++++++++++++++
    # kv_out_combined 形状是 [B, H, NUM_BLOCK + 1, D, E]，最后一个切片是更新后的 kv_history
    kv_history_out = kv_out_combined[:, :, -1, :, :].squeeze(2)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    os.makedirs("golden_data", exist_ok=True)
    
    np.save("golden_data/prefill_q.npy", q.cpu().numpy())
    np.save("golden_data/prefill_k.npy", k.cpu().numpy())
    np.save("golden_data/prefill_v.npy", v.cpu().numpy())
    np.save("golden_data/prefill_slope_rate.npy", slope_rate.cpu().numpy())
    np.save("golden_data/prefill_kv_history_in.npy", kv_history_in.cpu().numpy())
    np.save("golden_data/prefill_output_golden.npy", output.cpu().numpy())
    np.save("golden_data/prefill_kv_history_out_golden.npy", kv_history_out.cpu().numpy()) # 保存正确提取的值

    print("Prefill data saved.")
    print(f"Saved golden kv_history_out shape: {kv_history_out.shape}")


def generate_decode_data():
    print("--- Generating Decode Golden Data (using float32) ---")
    B, H, D = 4, 64, 128
    BLOCK_SIZE = 32
    torch.manual_seed(0)
    
    q = torch.randn(B, H, 1, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, 1, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, 1, D, dtype=torch.float32, device="cuda")
    kv_caches_in = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda")
    slope_rate = (torch.randn(H, dtype=torch.float32, device="cuda") * 0.1).abs()
    slot_idx = torch.arange(0, B, dtype=torch.int32, device="cuda")

    kv_caches_clone = kv_caches_in.clone()
    output = linear_decode_forward_triton(
        q, k, v, kv_caches_clone, slope_rate, slot_idx, BLOCK_SIZE
    )
    
    os.makedirs("golden_data", exist_ok=True)
    np.save("golden_data/decode_q.npy", q.cpu().numpy())
    np.save("golden_data/decode_k.npy", k.cpu().numpy())
    np.save("golden_data/decode_v.npy", v.cpu().numpy())
    np.save("golden_data/decode_kv_caches_in.npy", kv_caches_in.cpu().numpy())
    np.save("golden_data/decode_kv_caches_out_golden.npy", kv_caches_clone.cpu().numpy())
    np.save("golden_data/decode_slope_rate.npy", slope_rate.cpu().numpy())
    np.save("golden_data/decode_slot_idx.npy", slot_idx.cpu().numpy())
    np.save("golden_data/decode_output_golden.npy", output.cpu().numpy())
    print("Decode data saved.")

if __name__ == "__main__":
    generate_prefill_data()
    generate_decode_data()