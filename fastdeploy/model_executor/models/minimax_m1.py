# fastdeploy/model_executor/models/minimax_m1.py

from __future__ import annotations

import math
import re
from functools import partial

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
from fastdeploy.model_executor.models.model_base import (ModelForCasualLM,
                                                          ModelRegistry)
from fastdeploy.model_executor.utils import (default_weight_loader,
                                               process_weights_after_loading)
# 导入经过验证的 Triton 算子
from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import (
    lightning_attention_fd,
    linear_decode_forward_triton_fd,
)

class MiniMaxM1MLP(nn.Layer):
    """标准的 MLP 模块，用于 MoE 中的共享专家。"""
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
            with_bias=False
        )
        self.down_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=config.hidden_size,
            with_bias=False,
            reduce_results=reduce_results
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
        
        # 定义专家权重在 HuggingFace checkpoint 中的名字格式
        weight_key_map = {
            # FastDeploy FusedMoE 内部会将 w1 和 w3 映射到 up_gate_proj
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
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}", with_bias=False)
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
        # ... (此函数保持不变)
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
            attn_hidden, updated_kv_history = lightning_attention_fd(q, k, v, self.slope_rate, kv_history=kv_history)
            forward_meta.caches[self.layer_id][0] = updated_kv_history
            attn_hidden = attn_hidden.transpose([0, 2, 1, 3]).reshape([num_tokens, H * D])
        elif forward_meta.forward_mode.is_decode():
            B = num_tokens
            q = q.reshape([B, H, 1, D])
            k = k.reshape([B, H, 1, D])
            v = v.reshape([B, H, 1, D])
            kv_caches = forward_meta.caches[self.layer_id][0]
            slot_idx = forward_meta.block_tables[:, 0].squeeze(-1)
            attn_hidden = linear_decode_forward_triton_fd(q, k, v, kv_caches, self.slope_rate.squeeze(), slot_idx)
        else:
            raise NotImplementedError("Mixed mode is not supported for Linear Attention yet.")
            
        hidden_norm = self.norm(attn_hidden)
        gate = self.output_gate(hidden_states)
        gated_hidden = F.sigmoid(gate) * hidden_norm
        output = self.out_proj(gated_hidden.cast(hidden_states.dtype))
        return output

class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id
        self.postnorm = config.postnorm

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

        # MoE 块的前缀现在与 vLLM 和 HF checkpoint 对齐
        self.mlp = MiniMaxM1MoEBlock(fd_config, layer_id, prefix=f"{prefix}.block_sparse_moe")
        
        self.shared_moe = config.shared_intermediate_size > 0
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
        # 严格对齐 VLLM 的 DeepNorm + PostNorm 实现
        residual = hidden_states
        layernorm_output = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(forward_meta, layernorm_output)
        hidden_states = (residual * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)

        residual = hidden_states
        layernorm_output = self.post_attention_layernorm(hidden_states)
        
        moe_output = self.mlp(layernorm_output)
        
        if self.shared_moe:
            shared_mlp_output = self.shared_mlp(layernorm_output)
            coef = self.coefficient(layernorm_output.cast("float32"))
            coef = F.sigmoid(coef)
            mlp_output = (moe_output.cast("float32") * (1 - coef) + 
                          shared_mlp_output.cast("float32") * coef).cast(hidden_states.dtype)
        else:
            mlp_output = moe_output

        hidden_states = (residual * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        
        return hidden_states, None

@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        # 将 fd_config 保存为实例属性，以便子模块可以访问
        self.fd_config = fd_config
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

@ModelRegistry.register_model_class
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        config = self.fd_config.model_config
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim") and config.rotary_dim < config.head_dim:
            config.partial_rotary_factor = config.rotary_dim / config.head_dim
        
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, prefix="lm_head")

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        return self.model(ids_remove_padding, forward_meta)

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs):
        return self.lm_head(hidden_states)

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "w1", "gate"), ("up_gate_proj", "w3", "up"),
            ("shared_mlp.up_gate_proj", "shared_mlp.w1", "gate"),
            ("shared_mlp.up_gate_proj", "shared_mlp.w3", "up"),
        ]
        
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.num_local_experts,
            ckpt_gate_proj_name="w1.weight", ckpt_up_proj_name="w3.weight",
            ckpt_down_proj_name="w2.weight",
            param_gate_up_proj_name="mlp.experts.up_gate_proj_",
            param_down_proj_name="mlp.experts.down_proj_",
            ckpt_expert_key_name="block_sparse_moe.experts"
        )
        
        params_dict = dict(self.named_parameters())
        
        for loaded_weight_name, loaded_weight in weights_iterator:
            model_param_name = loaded_weight_name
            found = False

            # 统一将 HF 的 MoE 块名转为 FD 内部名
            if "block_sparse_moe.experts" in loaded_weight_name:
                for p_name, ckpt_pattern, expert_id, shard_id in expert_params_mapping:
                    # 构建精确的 checkpoint 权重名模式
                    full_ckpt_pattern = ckpt_pattern.replace("{}", str(expert_id))
                    if full_ckpt_pattern in loaded_weight_name:
                        # 替换成 FD 内部参数名
                        model_param_name = loaded_weight_name.replace(full_ckpt_pattern, p_name)
                        if model_param_name in params_dict:
                            param = params_dict[model_param_name]
                            param.weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                            found = True
                            break
            if found: continue

            # 匹配标准融合层
            for p_name, ckpt_name, shard_id in stacked_params_mapping:
                # 构造精确匹配模式，避免误匹配 (e.g., matching ".w1." not just "w1")
                if f".{ckpt_name}." in loaded_weight_name:
                    model_param_name = loaded_weight_name.replace(ckpt_name, p_name)
                    if model_param_name in params_dict:
                        param = params_dict[model_param_name]
                        getattr(param, "weight_loader", default_weight_loader(self.fd_config))(param, loaded_weight, shard_id)
                        found = True
                        break
            if found: continue
            
            # 匹配剩余的单体权重
            name_map = {
                ".w2.weight": ".down_proj.weight",
                ".shared_mlp.w2.weight": ".shared_mlp.down_proj.weight"
            }
            for ckpt_suffix, fd_suffix in name_map.items():
                if loaded_weight_name.endswith(ckpt_suffix):
                    model_param_name = loaded_weight_name.replace(ckpt_suffix, fd_suffix)
                    break

            if model_param_name in params_dict:
                param = params_dict[model_param_name]
                getattr(param, "weight_loader", default_weight_loader(self.fd_config))(param, loaded_weight)
            elif "rotary_emb.inv_freq" not in loaded_weight_name:
                logger.warning(f"Weight {loaded_weight_name} (mapped to {model_param_name}) not found in model.")

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method.")

@ModelRegistry.register_pretrained_model
class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig

    @classmethod
    def arch_name(cls):
        return "MiniMaxM1ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.warning("Tensor parallel mappings for MiniMax-M1 are placeholders and not yet implemented.")
        return {}