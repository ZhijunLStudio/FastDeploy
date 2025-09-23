# fastdeploy/model_executor/models/minimax_m1.py

from __future__ import annotations

import re
from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
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
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
from fastdeploy.model_executor.models.utils import (
    LayerIdPlaceholder as layerid, WeightMeta)
from fastdeploy.model_executor.utils import default_weight_loader, process_weights_after_loading


class MiniMaxM1MLP(nn.Layer):
    """标准的 MLP 模块，用于 MoE 中的共享专家或非 MoE 层的 MLP。"""
    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True
    ):
        super().__init__()
        config = fd_config.model_config
        # vllm中 w1和w3合并为gate_up_proj
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
        )
        # vllm中 w2 对应 down_proj
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
            output_size=config.num_local_experts, # vllm 中是 num_total_experts
            with_bias=False,
            weight_dtype="float32",
        )
        
        # 映射 checkpoint 中的专家权重名到 FastDeploy 参数名
        # vllm中 w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj
        weight_key_map = {
            "gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.w1.weight",
            "up_proj_expert_weight_key": f"{prefix}.experts.{{}}.w3.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.w2.weight",
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
        # FastDeploy 的 FusedMoE.apply 需要传入 gate 层
        out = self.experts(hidden_states, self.gate)
        return out


class MiniMaxM1StandardAttention(nn.Layer):
    """标准注意力 (GQA + Partial RoPE)。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        # QKVParallelLinear的 weight_loader 能够处理从 q_proj, k_proj, v_proj 分开加载权重
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False
        )
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        atten_out = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        output = self.o_proj(atten_out)
        return output


class MiniMaxM1LinearAttention(nn.Layer):
    """线性注意力模块的占位符。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        # 根据vllm实现，这里也需要相应的线性层
        config = fd_config.model_config
        hidden_inner_size = config.head_dim * config.num_attention_heads
        
        self.qkv_proj = RowParallelLinear(
            fd_config, prefix=f"{prefix}.qkv_proj",
            input_size=config.hidden_size,
            output_size=hidden_inner_size * 3, with_bias=False)
            
        self.output_gate = RowParallelLinear(
            fd_config, prefix=f"{prefix}.output_gate",
            input_size=config.hidden_size,
            output_size=hidden_inner_size, with_bias=False)
            
        self.out_proj = RowParallelLinear(
            fd_config, prefix=f"{prefix}.out_proj",
            input_size=hidden_inner_size,
            output_size=config.hidden_size, with_bias=False)
            
        self.norm = RMSNorm(fd_config, hidden_size=hidden_inner_size, eps=1e-5, prefix=f"{prefix}.norm")

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        # vllm中使用了名为 `lightning_attention` 的自定义 Triton Kernel。
        # 在 FastDeploy 中需要移植或实现对应的底层算子。
        # 在此之前，我们先构建好 Python 层的逻辑结构。
        raise NotImplementedError(
            "MiniMaxM1LinearAttention requires a custom Triton/CUTLASS kernel "
            "similar to vLLM's 'lightning_attention', which is not yet implemented in FastDeploy."
        )


class MiniMaxM1DecoderLayer(nn.Layer):
    """核心解码器层，动态选择注意力类型并应用 DeepNorm。"""
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id
        self.postnorm = config.postnorm

        # 动态选择注意力模块并设置对应的 DeepNorm 参数
        attn_type = config.attn_type_list[layer_id]
        attn_prefix = f"{prefix}.self_attn"
        if attn_type == 1: # Standard Attention
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        elif attn_type == 0: # Linear Attention
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            raise ValueError(f"Unknown attention type: {attn_type} for layer {layer_id}")

        # MoE 模块
        self.mlp = MiniMaxM1MoEBlock(fd_config, layer_id, prefix=f"{prefix}.block_sparse_moe")
        
        # 共享专家模块
        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(
                fd_config, 
                config.shared_intermediate_size, 
                prefix=f"{prefix}.shared_mlp"
            )
            self.coefficient = ReplicatedLinear(
                fd_config,
                prefix=f"{prefix}.coefficient",
                input_size=config.hidden_size,
                output_size=1,
                with_bias=False,
                weight_dtype="float32",
            )
        
        # MLP 的 DeepNorm 参数
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta

        # 归一化层
        self.input_layernorm = RMSNorm(fd_config, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, prefix=f"{prefix}.post_attention_layernorm")

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: paddle.Tensor | None):
        # 严格遵循 vllm 的 DeepNorm + PostNorm 实现
        
        # 1. 注意力块
        # Post-Norm: 对输入进行归一化，并将归一化后的结果作为残差
        residual = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(forward_meta, residual)
        # DeepNorm: 使用原始 hidden_states 进行残差连接
        hidden_states = (hidden_states * self.layernorm_attention_alpha) + \
                        (attn_output * self.layernorm_attention_beta)

        # 2. MLP/MoE 块
        # Post-Norm: 对注意力块的输出进行归一化
        residual = self.post_attention_layernorm(hidden_states)
        
        # MoE 计算
        moe_output = self.mlp(residual)
        
        # 共享专家计算 (如果存在)
        if self.shared_moe:
            shared_mlp_output = self.shared_mlp(residual)
            coef = self.coefficient(residual.cast("float32"))
            coef = F.sigmoid(coef)
            mlp_output = (moe_output.cast("float32") * (1 - coef) + 
                          shared_mlp_output.cast("float32") * coef).cast(hidden_states.dtype)
        else:
            mlp_output = moe_output

        # DeepNorm: 使用第二块的输入 (hidden_states) 进行残差连接
        hidden_states = (hidden_states * self.layernorm_mlp_alpha) + \
                        (mlp_output * self.layernorm_mlp_beta)
        
        return hidden_states, None # vllm 中 residual 在层间不传递，因此返回 None


@support_graph_optimization
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
        # 设置部分旋转位置编码因子
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim"):
            if config.rotary_dim < config.head_dim:
                config.partial_rotary_factor = config.rotary_dim / config.head_dim

        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, prefix="lm_head")

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.model(ids_remove_padding, forward_meta)
        return hidden_states

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs):
        return self.lm_head(hidden_states)

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        """
        使用 FastDeploy 的 default_loader_v1 模式加载权重。
        此方法定义了从 checkpoint 权重名到模型参数名的映射关系。
        """
        
        # 1. 映射标准（非 MoE 专家）层的权重
        stacked_params_mapping = [
            # (FastDeploy参数名, checkpoint权重名, 分片ID)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # MiniMaxM1 MLP/MoE 使用 w1, w2, w3 命名
            ("up_gate_proj", "w1", "gate"), # w1 -> gate_proj
            ("up_gate_proj", "w3", "up"),   # w3 -> up_proj
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]
        
        # 2. 映射 MoE 专家层的权重
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.num_local_experts,
            ckpt_gate_proj_name="w1",
            ckpt_up_proj_name="w3",
            ckpt_down_proj_name="w2",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
            ckpt_expert_key_name="experts",
        )
        
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))

        # 3. 遍历 checkpoint 中的权重并加载
        for loaded_weight_name, loaded_weight in weights_iterator:
            # vllm checkpoint 的前缀是 model.
            model_param_name = loaded_weight_name
            
            # 首先尝试匹配标准层
            found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                # 排除 MoE 专家权重，它们由 expert_params_mapping 处理
                if "block_sparse_moe.experts" in loaded_weight_name:
                    continue
                
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name in params_dict:
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight, shard_id)
                    found = True
                    break
            if found:
                continue

            # 然后尝试匹配 MoE 专家层
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name in params_dict:
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    found = True
                    break
            if found:
                continue
                
            # 最后，处理剩余的权重（如 layernorm, o_proj, down_proj, coefficient 等）
            if model_param_name.replace("w2", "down_proj") in params_dict:
                 model_param_name = model_param_name.replace("w2", "down_proj")

            if model_param_name in params_dict:
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(weight|bias)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)
            
    def set_state_dict(self, state_dict):
        # 在新的加载器模式下，这个函数通常不需要了，因为权重是通过 load_weights 加载的。
        # 如果需要支持旧的 set_state_dict 模式，需要在这里实现完整的加载逻辑。
        # 但根据你的需求，我们遵循 v1 加载器模式。
        raise NotImplementedError("MiniMax-M1 uses the new `load_weights` method with default_loader_v1.")


# 注册模型以便 FastDeploy 和 PaddleNLP 能够找到它
# @ModelRegistry.register_pretrained_model
class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig

    @classmethod
    def arch_name(cls):
        return "MiniMaxM1ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        # 定义张量并行切分规则
        from paddleformers.transformers.conversion_utils import split_or_merge_func
        
        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        # ... (此处省略详细的 TP Mappings 实现，可以参考 GLM-4 或 Qwen3) ...
        # 这是一个复杂的步骤，需要根据每个线性层的切分方式（行切分或列切分）来定义
        # 对于 MiniMax-M1，切分规则与 Qwen/LLaMA 类似
        
        logger.warning("Tensor parallel mappings for MiniMax-M1 are not fully implemented. Using a placeholder.")
        return {}