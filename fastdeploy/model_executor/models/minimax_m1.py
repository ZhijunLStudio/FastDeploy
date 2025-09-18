# fastdeploy/model_executor/models/minimax_m1.py

from __future__ import annotations
import paddle
from paddle import nn
import paddle.nn.functional as F

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM

# --- 模块 1: 标准 MLP (用于 MoE 中的共享专家) ---
class MiniMaxM1MLP(nn.Layer):
    """标准的 MLP 模块，使用 SwiGLU 激活函数。"""
    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.gate_up_proj",
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
        )
        self.act_fn = SiluAndMul()

    def load_state_dict(self, state_dict):
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

# --- 模块 1.5: MoE Block (遵循 qwen3moe 的新架构) ---
class MiniMaxM1MoEBlock(nn.Layer):
    """一个完整的 MoE 模块，包含独立的路由器(gate)和专家组(experts)。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            with_bias=False,
            weight_dtype="float32",
        )
        
        weight_key_map = {
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
        }

        self.experts = FusedMoE(
            fd_config,
            moe_intermediate_size=config.intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

    def load_state_dict(self, state_dict):
        self.gate.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)
        
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        # FusedMoE 的新接口需要传入 gate 实例
        out = self.experts(hidden_states, self.gate)
        return out

# --- 模块 2: 标准注意力 (GQA + Partial RoPE) ---
class MiniMaxM1StandardAttention(nn.Layer):
    """处理 attn_type: 1 的标准注意力层。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=config.num_attention_heads * config.head_dim,
            output_size=config.hidden_size,
        )
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.attn.load_state_dict(state_dict)

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        atten_out = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        output = self.o_proj(atten_out)
        return output

# --- 模块 3: 线性注意力 (当前留空) ---
class MiniMaxM1LinearAttention(nn.Layer):
    """处理 attn_type: 0 的线性注意力层。这是后续开发的核心。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        # TODO: 第二阶段开发任务
        pass
    
    def load_state_dict(self, state_dict):
        # TODO: 第二阶段开发任务
        pass
        
    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        # TODO: 第二阶段开发任务
        raise NotImplementedError(f"Linear Attention (attn_type=0) for layer {self.layer_id} is not yet implemented.")

# --- 模块 4: 核心解码器层 ---
class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id

        # a. 动态选择注意力模块并设置对应的 DeepNorm 参数
        attn_type = config.attn_type_list[layer_id]
        if attn_type == 1:
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        elif attn_type == 0:
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            raise ValueError(f"Unknown attention type: {attn_type} for layer {layer_id}")

        # b. MoE/MLP 模块 (遵循新架构)
        self.mlp = MiniMaxM1MoEBlock(fd_config, layer_id, prefix=f"{prefix}.mlp")
        
        # c. 共享专家模块
        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(fd_config, config.shared_intermediate_size, prefix=f"{prefix}.shared_mlp")
            self.coefficient = ReplicatedLinear(
                fd_config,
                prefix=f"{prefix}.coefficient",
                input_size=config.hidden_size,
                output_size=1,
                with_bias=False,
                weight_dtype="float32",
            )
        
        # d. MLP 的 DeepNorm 参数
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta
        self.postnorm = config.postnorm

        # e. 归一化层
        self.input_layernorm = RMSNorm(fd_config, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, prefix=f"{prefix}.post_attention_layernorm")
        
    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        if self.shared_moe:
            self.shared_mlp.load_state_dict(state_dict)
            self.coefficient.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: paddle.Tensor | None):
        layernorm_input = hidden_states
        layernorm_output = self.input_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input
        attn_output = self.self_attn(forward_meta, layernorm_output)
        hidden_states = (residual * self.layernorm_attention_beta + attn_output * self.layernorm_attention_alpha)

        layernorm_input = hidden_states
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input
        
        moe_output = self.mlp(layernorm_output)
        if self.shared_moe:
            shared_mlp_output = self.shared_mlp(layernorm_output)
            coef, _ = self.coefficient(layernorm_output.cast("float32"))
            coef = F.sigmoid(coef)
            mlp_output = (moe_output.cast("float32") * (1 - coef) + 
                          shared_mlp_output.cast("float32") * coef).cast(hidden_states.dtype)
        else:
            mlp_output = moe_output

        hidden_states = (residual * self.layernorm_mlp_beta + mlp_output * self.layernorm_mlp_alpha)
        
        return hidden_states, None

# --- 5. 顶层模型 ---
class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(fd_config, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.LayerList(
            [
                MiniMaxM1DecoderLayer(fd_config, i, prefix=f"{prefix}.layers.{i}")
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(fd_config, prefix=f"{prefix}.norm")

    def load_state_dict(self, state_dict):
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.config.num_hidden_layers):
            self.layers[i].load_state_dict(state_dict)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(ids_remove_padding)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(forward_meta, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        
        config = self.fd_config.model_config
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim"):
            if config.rotary_dim < config.head_dim:
                config.partial_rotary_factor = config.rotary_dim / config.head_dim
        
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, prefix="lm_head")

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.model(ids_remove_padding, forward_meta)
        return hidden_states

    def compute_logits(self, hidden_states: paddle.Tensor):
        return self.lm_head(hidden_states)