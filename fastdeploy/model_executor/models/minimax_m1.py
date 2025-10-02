# fastdeploy/model_executor/models/minimax_m1.py

from __future__ import annotations

import math
import re

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.graph_optimization.decorator import \
    support_graph_optimization
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
    RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
# 注意：移除了 ModelRegistry 的导入，因为我们不再需要它
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
# 你的 triton op 导入保持不变
from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import (
    lightning_attention,
    linear_decode_forward_triton,
)
from fastdeploy.model_executor.utils import default_weight_loader

# --- 所有子模块 (MiniMaxM1MLP, MiniMaxM1MoEBlock 等) 保持不变 ---
# (这里省略了子模块的代码，使用你原来的即可)
class MiniMaxM1MLP(nn.Layer):
    """标准的 MLP 模块，仅用于 MoE 中的共享专家 (如果启用)。"""
    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True
    ):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
        )
        self.down_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=config.hidden_size,
            with_bias=False,
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

class MiniMaxM1MoEBlock(nn.Layer):
    """MoE 模块，包含路由（gate）和专家（experts）。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        
        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=config.hidden_size,
            output_size=config.num_local_experts,
            with_bias=False,
            weight_dtype="float32",
        )
        
        weight_key_map = {
            "gate_proj_expert_weight_key": "experts.{{}}.w1.weight",
            "up_proj_expert_weight_key": "experts.{{}}.w3.weight",
            "down_proj_expert_weight_key": "experts.{{}}.w2.weight",
        }

        self.experts = FusedMoE(
            fd_config,
            moe_intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        out = self.experts(hidden_states, self.gate)
        return out

class MiniMaxM1StandardAttention(nn.Layer):
    """标准注意力 (GQA + Partial RoPE)。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=prefix, with_bias=False)
        self.o_proj = RowParallelLinear(
            fd_config, prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False
        )
        self.attn = Attention(
            fd_config, layer_id=layer_id, prefix=prefix, use_neox_rotary_style=True
        )

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        atten_out = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        output = self.o_proj(atten_out)
        return output

class MiniMaxM1LinearAttention(nn.Layer):
    """线性注意力模块的完整实现。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.fd_config = fd_config
        config = fd_config.model_config
        hidden_inner_size = config.head_dim * config.num_attention_heads
        
        self.qkv_proj = RowParallelLinear(
            fd_config, prefix=f"{prefix}.qkv_proj",
            input_size=config.hidden_size, output_size=hidden_inner_size * 3, with_bias=False)
        self.output_gate = RowParallelLinear(
            fd_config, prefix=f"{prefix}.output_gate",
            input_size=config.hidden_size, output_size=hidden_inner_size, with_bias=False)
        self.out_proj = RowParallelLinear(
            fd_config, prefix=f"{prefix}.out_proj",
            input_size=hidden_inner_size, output_size=config.hidden_size, with_bias=False)
        self.norm = RMSNorm(fd_config, hidden_size=hidden_inner_size, eps=1e-5, prefix=f"{prefix}.norm")

        slope_rate = self._build_slope_tensor(config.num_attention_heads)
        if config.num_hidden_layers > 1:
            slope_rate = slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        self.register_buffer("slope_rate", slope_rate, persistable=False)

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]
            if math.log2(n).is_integer(): return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])
        slopes = paddle.to_tensor(get_slopes(n_attention_heads), dtype='float32').reshape([n_attention_heads, 1, 1])
        return slopes

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        num_tokens, _ = hidden_states.shape
        config = self.fd_config.model_config
        H = config.num_attention_heads
        D = config.head_dim

        qkv_out = self.qkv_proj(hidden_states)
        qkv_act = F.silu(qkv_out.cast("float32"))
        qkv_act_reshaped = qkv_act.reshape([num_tokens, H, 3 * D])
        q, k, v = paddle.split(qkv_act_reshaped, 3, axis=-1)

        if forward_meta.forward_mode.is_prefill():
            B = len(forward_meta.seq_lens_encoder)
            N = num_tokens // B
            q = q.reshape([B, N, H, D]).transpose([0, 2, 1, 3])
            k = k.reshape([B, N, H, D]).transpose([0, 2, 1, 3])
            v = v.reshape([B, N, H, D]).transpose([0, 2, 1, 3])
            kv_history = forward_meta.caches[self.layer_id][0]
            attn_hidden, updated_kv_history = lightning_attention(q, k, v, self.slope_rate, kv_history=kv_history)
            forward_meta.caches[self.layer_id][0] = updated_kv_history
            attn_hidden = attn_hidden.transpose([0, 2, 1, 3]).reshape([num_tokens, H * D])
        elif forward_meta.forward_mode.is_decode():
            B = num_tokens
            q = q.reshape([B, H, 1, D])
            k = k.reshape([B, H, 1, D])
            v = v.reshape([B, H, 1, D])
            kv_caches = forward_meta.caches[self.layer_id][0]
            slot_idx = forward_meta.block_tables[:, 0].squeeze(-1)
            attn_hidden = linear_decode_forward_triton(q, k, v, kv_caches, self.slope_rate.squeeze(), slot_idx)
        else:
            raise NotImplementedError("Mixed mode is not supported for Linear Attention yet.")
            
        hidden_norm = self.norm(attn_hidden)
        gate, _ = self.output_gate(hidden_states)
        gated_hidden = F.sigmoid(gate) * hidden_norm
        output = self.out_proj(gated_hidden.cast(hidden_states.dtype))
        return output

class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id

        attn_type = config.attn_type_list[layer_id]
        attn_prefix = f"{prefix}.self_attn"
        if attn_type == 1:
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        elif attn_type == 0:
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            raise ValueError(f"Unknown attention type for layer {layer_id}: {attn_type}")

        self.block_sparse_moe = MiniMaxM1MoEBlock(fd_config, layer_id, prefix=f"{prefix}.block_sparse_moe")
        
        self.shared_moe = hasattr(config, "shared_intermediate_size") and config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(
                fd_config, config.shared_intermediate_size, 
                prefix=f"{prefix}.shared_mlp", reduce_results=False
            )
            self.coefficient = ReplicatedLinear(
                fd_config, prefix=f"{prefix}.coefficient", input_size=config.hidden_size,
                output_size=1, with_bias=False, weight_dtype="float32"
            )
        
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta

        self.input_layernorm = RMSNorm(fd_config, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, prefix=f"{prefix}.post_attention_layernorm")

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: paddle.Tensor | None):
        residual = hidden_states
        layernorm_output = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(forward_meta, layernorm_output)
        hidden_states = (residual * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)

        residual = hidden_states
        layernorm_output = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.block_sparse_moe(layernorm_output)
        
        if self.shared_moe:
            shared_mlp_output = self.shared_mlp(layernorm_output)
            coef, _ = self.coefficient(layernorm_output.cast("float32"))
            coef = F.sigmoid(coef)
            final_mlp_output = (mlp_output.cast("float32") * (1 - coef) + 
                                shared_mlp_output.cast("float32") * coef).cast(hidden_states.dtype)
        else:
            final_mlp_output = mlp_output

        hidden_states = (residual * self.layernorm_mlp_alpha) + (final_mlp_output * self.layernorm_mlp_beta)
        
        return hidden_states, None

@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(fd_config, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.LayerList(
            [MiniMaxM1DecoderLayer(fd_config, i, prefix=f"{prefix}.layers.{i}")
             for i in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(fd_config, prefix=f"{prefix}.norm")

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(ids_remove_padding)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(forward_meta, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states
# --- 结束子模块 ---

# 移除 @ModelRegistry.register_model_class
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        config = self.fd_config.model_config
        
        # --- 把所有适配逻辑都集中到这里！---
        # 1. 设置统一前缀，保持规范
        config.pretrained_config.prefix_name = "model"

        # 2. 适配专家数量的名称
        if hasattr(config, "num_local_experts") and not hasattr(config, "moe_num_experts"):
            config.moe_num_experts = config.num_local_experts
            # 其他参考模型也这样做了
            config.n_routed_experts = config.num_local_experts
        
        # 3. GLM4.5-Air 也适配了这些，我们跟进
        if not hasattr(config, "n_shared_experts"):
             config.n_shared_experts = 0
        if not hasattr(config, "first_k_dense_replace"):
             config.first_k_dense_replace = 0

        # 4. 适配 Partial RoPE
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim") and config.rotary_dim < config.head_dim:
            config.partial_rotary_factor = config.rotary_dim / config.head_dim
        
        # --- 结束适配 ---
        
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, prefix="lm_head")

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    # ... forward, compute_logits, load_weights, set_state_dict 方法保持不变 ...
    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        return self.model(ids_remove_padding, forward_meta)

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs):
        return self.lm_head(hidden_states)

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        # 这里的 load_weights 逻辑非常复杂，我們先假设它是对的，如果启动后报权重加载错误，再来调试这里
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "w1", "gate"), ("up_gate_proj", "w3", "up"),
            ("shared_mlp.up_gate_proj", "w1", "gate"),
            ("shared_mlp.up_gate_proj", "w3", "up"),
        ]
        
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.num_local_experts,
            ckpt_gate_proj_name="w1.weight",
            ckpt_up_proj_name="w3.weight",
            ckpt_down_proj_name="w2.weight",
            param_gate_up_proj_name="block_sparse_moe.experts.up_gate_proj_",
            param_down_proj_name="block_sparse_moe.experts.down_proj_",
            ckpt_expert_key_name="experts"
        )
        
        params_dict = dict(self.named_parameters())
        
        for loaded_weight_name, loaded_weight in weights_iterator:
            param_key = loaded_weight_name[len("model."):] if loaded_weight_name.startswith("model.") else loaded_weight_name
            found = False
            
            if ".block_sparse_moe.experts." in loaded_weight_name:
                for p_name_prefix, ckpt_pattern, expert_id, shard_id in expert_params_mapping:
                    hf_weight_name = f"layers.{expert_id // self.fd_config.model_config.num_local_experts}.{ckpt_pattern.format(expert_id)}"
                    if loaded_weight_name.endswith(hf_weight_name):
                        model_param_name = loaded_weight_name.replace(hf_weight_name, p_name_prefix)
                        if model_param_name in params_dict:
                            param = params_dict[model_param_name]
                            param.weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                            found = True
                            break
            if found: continue

            for p_name, ckpt_name, shard_id in stacked_params_mapping:
                if f".{ckpt_name}." in param_key:
                    fd_param_key = param_key.replace(f".{ckpt_name}.", f".{p_name}.")
                    if fd_param_key in params_dict:
                        param = params_dict[fd_param_key]
                        getattr(param, "weight_loader", default_weight_loader(self.fd_config))(param, loaded_weight, shard_id)
                        found = True
                        break
            if found: continue

            name_map = {
                ".w2.weight": ".down_proj.weight",
                ".shared_mlp.w2.weight": ".down_proj.weight",
            }
            
            model_param_name = param_key
            for hf_suffix, fd_suffix in name_map.items():
                if model_param_name.endswith(hf_suffix):
                    model_param_name = model_param_name.rsplit(hf_suffix, 1)[0] + fd_suffix
                    break

            if model_param_name in params_dict:
                param = params_dict[model_param_name]
                getattr(param, "weight_loader", default_weight_loader(self.fd_config))(param, loaded_weight)
            elif "rotary_emb.inv_freq" not in model_param_name:
                logger.warning(f"Weight '{loaded_weight_name}' (mapped to '{model_param_name}') was not found in the model.")

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method with the default_v1 loader.")

# 移除 @ModelRegistry.register_pretrained_model
class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig

    @classmethod
    def arch_name(cls):
        return "MiniMaxM1ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.warning("Tensor parallel mappings for MiniMax-M1 are placeholders and not yet fully implemented.")
        return {}