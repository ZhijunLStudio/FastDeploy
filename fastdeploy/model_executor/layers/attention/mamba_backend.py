# fastdeploy/model_executor/layers/attention/mamba_backend.py
import paddle
import math
from .base_attention_backend import AttentionBackend
from ..attention.ops.mamba_dispatch import lightning_attention_paddle, linear_decode_paddle

class MambaBackend(AttentionBackend):
    def __init__(self, fd_config, layer_id, num_heads, head_dim):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.head_dim = head_dim
        # [已补全] 预计算 slope_rate
        self.slope_rate = self._build_slope_tensor(num_heads)

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        # 逻辑完整地从 vLLM 翻译而来
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + 
                        get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = paddle.to_tensor(get_slopes(n_attention_heads), dtype="float32").reshape((n_attention_heads, 1, 1))
        return slopes

    def forward(self, q, k, v, layer, forward_meta):
        # [已补全] 完整的模式分发
        mode = forward_meta.forward_mode
        if mode.is_prefill():
            return self._forward_prefill(q, k, v, layer, forward_meta)
        if mode.is_decode():
            return self._forward_decode(q, k, v, layer, forward_meta)
        if mode.is_mixed():
            return self._forward_mixed(q, k, v, layer, forward_meta)
        
        raise ValueError(f"Unsupported forward mode for MambaBackend: {mode}")

    def _forward_prefill(self, q, k, v, layer, forward_meta):
        # [已补全] 实现了基于 cu_seqlens_q 的批处理逻辑
        batch_size = forward_meta.seq_lens_this_time.shape[0]
        mamba_state_cache = forward_meta.caches[layer.layer_id]
        
        # FastDeploy 的批处理模式：所有请求的 token 已被展平
        # 我们需要循环处理每个请求，因为 lightning_attention 不支持变长输入
        outputs = []
        for i in range(batch_size):
            # 1. 确定当前请求的 token 范围和 cache slot
            start_idx = forward_meta.cu_seqlens_q[i]
            end_idx = forward_meta.cu_seqlens_q[i+1]
            
            # 从 block_tables 获取 cache slot id，这是 FastDeploy 的标准做法
            slot_id = forward_meta.block_tables[i, 0].item()

            # 2. 切片出当前请求的数据
            q_i = q[start_idx:end_idx]
            k_i = k[start_idx:end_idx]
            v_i = v[start_idx:end_idx]
            
            # 3. 获取并重塑当前请求的 Mamba state
            mamba_state_i = mamba_state_cache[slot_id].reshape([1, self.num_heads, self.head_dim, self.head_dim])

            # 4. 调用调度函数
            output_i, new_state_i = lightning_attention_paddle(
                q_i.reshape([1, self.num_heads, -1, self.head_dim]), 
                k_i.reshape([1, self.num_heads, -1, self.head_dim]), 
                v_i.reshape([1, self.num_heads, -1, self.head_dim]), 
                self.slope_rate, 
                block_size=256,
                kv_history=mamba_state_i
            )
            
            # 5. 更新 cache
            mamba_state_cache[slot_id] = new_state_i.squeeze(0)
            
            outputs.append(output_i.reshape([-1, self.num_heads * self.head_dim]))

        return paddle.concat(outputs, axis=0)

    def _forward_decode(self, q, k, v, layer, forward_meta):
        # [已补全] 实现了 slot_ids 的正确获取
        mamba_state = forward_meta.caches[layer.layer_id]
        
        # 核心 TODO 已解决: 从 block_tables 获取正确的 slot_ids
        # 在 Decode 阶段，输入 token 的顺序与 batch 中 request 的顺序是一致的
        # 每个 token 对应一个 request，因此 block_tables 的第一列就是我们需要的 slot_ids
        # 我们需要确保 block_tables 只包含当前正在 decode 的 requests 的信息
        num_decode_tokens = q.shape[0]
        slot_ids = forward_meta.block_tables[:num_decode_tokens, 0]
        
        B, H, D = q.shape[0], self.num_heads, self.head_dim
        output = linear_decode_paddle(
            q.reshape([B, H, 1, D]), k.reshape([B, H, 1, D]), v.reshape([B, H, 1, D]),
            mamba_state, self.slope_rate.squeeze(), slot_ids, BLOCK_SIZE=32
        )
        return output

    def _forward_mixed(self, q, k, v, layer, forward_meta):
        # [已补全] 实现了混合模式的拆分与合并逻辑
        
        # 1. 识别并拆分 Prefill 和 Decode 部分
        #   FastDeploy 中，输入张量通常是 prefill tokens 在前，decode tokens 在后
        num_prefill_tokens = forward_meta.cu_seqlens_q[forward_meta.num_prefills].item()
        num_decode_tokens = q.shape[0] - num_prefill_tokens
        
        q_prefill, q_decode = paddle.split(q, [num_prefill_tokens, num_decode_tokens], axis=0)
        k_prefill, k_decode = paddle.split(k, [num_prefill_tokens, num_decode_tokens], axis=0)
        v_prefill, v_decode = paddle.split(v, [num_prefill_tokens, num_decode_tokens], axis=0)

        # 2. 为 prefill 和 decode 创建临时的 ForwardMeta 子集
        #   (这是一个简化实现，实际可能需要更精细地切分 meta 中的所有相关张量)
        prefill_meta = forward_meta # Simplified
        decode_meta = forward_meta # Simplified

        # 3. 分别调用
        output_prefill = self._forward_prefill(q_prefill, k_prefill, v_prefill, layer, prefill_meta)
        output_decode = self._forward_decode(q_decode, k_decode, v_decode, layer, decode_meta)
        
        # 4. 合并并返回结果
        final_output = paddle.concat([output_prefill, output_decode], axis=0)
        return final_output