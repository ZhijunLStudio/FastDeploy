import unittest
import numpy as np
import paddle
import os

from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import (
    lightning_attention,
    linear_decode_forward_triton,
)

class TestMiniMaxMambaOps(unittest.TestCase):
    
    def setUp(self):
        self.device = paddle.CUDAPlace(0)
        self.golden_data_path = "./golden_data" # 建议使用相对路径

    def load_npy(self, filename: str):
        full_path = os.path.join(self.golden_data_path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Golden data file not found at: {full_path}. "
                "Please run `generate_golden_data.py` first."
            )
        # 显式转换为 float32，确保与生成数据时的精度一致
        return paddle.to_tensor(np.load(full_path), place=self.device, dtype='float32')

    def test_prefill_lightning_attention(self):
        print("\n--- Testing Prefill (lightning_attention) ---")
        q = self.load_npy("prefill_q.npy")
        k = self.load_npy("prefill_k.npy")
        v = self.load_npy("prefill_v.npy")
        slope_rate = self.load_npy("prefill_slope_rate.npy")
        kv_history_in = self.load_npy("prefill_kv_history_in.npy")
        
        output_golden = self.load_npy("prefill_output_golden.npy")
        kv_history_out_golden = self.load_npy("prefill_kv_history_out_golden.npy")

        # 确认输入形状
        print(f"DEBUG (Prefill): Input q shape: {q.shape}")
        print(f"DEBUG (Prefill): Input kv_history_in shape: {kv_history_in.shape}")

        output_fd, kv_history_out_fd = lightning_attention(
            q, k, v, slope_rate, kv_history=kv_history_in
        )
        
        # 确认输出形状
        print(f"DEBUG (Prefill): FD output shape: {output_fd.shape}, Golden output shape: {output_golden.shape}")
        print(f"DEBUG (Prefill): FD kv_history_out shape: {kv_history_out_fd.shape}, Golden kv_history_out shape: {kv_history_out_golden.shape}")


        rtol, atol = 1e-4, 1e-4
        
        np.testing.assert_allclose(
            output_fd.cpu().numpy(),
            output_golden.cpu().numpy(),
            rtol=rtol, atol=atol, err_msg="Prefill output mismatch!"
        )
        print("Prefill output validation PASSED.")
        
        np.testing.assert_allclose(
            kv_history_out_fd.cpu().numpy(),
            kv_history_out_golden.cpu().numpy(),
            rtol=rtol, atol=atol, err_msg="Prefill kv_history_out mismatch!"
        )
        print("Prefill kv_history_out validation PASSED.")
        
    def test_decode_linear_attention(self):
        print("\n--- Testing Decode (linear_decode_forward_triton) ---")
        q = self.load_npy("decode_q.npy")
        k = self.load_npy("decode_k.npy")
        v = self.load_npy("decode_v.npy")
        kv_caches_in = self.load_npy("decode_kv_caches_in.npy")
        slope_rate = self.load_npy("decode_slope_rate.npy")
        slot_idx = self.load_npy("decode_slot_idx.npy")

        output_golden = self.load_npy("decode_output_golden.npy")
        kv_caches_out_golden = self.load_npy("decode_kv_caches_out_golden.npy")
        
        print(f"DEBUG (Decode): Input q shape: {q.shape}")

        kv_caches_fd = kv_caches_in.clone()
        output_fd = linear_decode_forward_triton(
            q, k, v, kv_caches_fd, slope_rate, slot_idx, BLOCK_SIZE=32
        )

        print(f"DEBUG (Decode): FD output shape: {output_fd.shape}, Golden output shape: {output_golden.shape}")

        rtol, atol = 1e-4, 1e-4

        np.testing.assert_allclose(
            output_fd.cpu().numpy(),
            output_golden.cpu().numpy(),
            rtol=rtol, atol=atol, err_msg="Decode output mismatch!"
        )
        print("Decode output validation PASSED.")

        np.testing.assert_allclose(
            kv_caches_fd.cpu().numpy(),
            kv_caches_out_golden.cpu().numpy(),
            rtol=rtol, atol=atol, err_msg="Decode kv_caches_out mismatch!"
        )
        print("Decode kv_caches_out validation PASSED.")


if __name__ == "__main__":
    unittest.main()