# /home/aistudio/work/FastDeploy/fastdeploy/model_executor/models/kimi_k2.py
from __future__ import annotations
import re
import math
import paddle
import paddle.nn as nn
from paddleformers.utils.log import logger

# --- Imports ---
from .deepseek_v3 import (
    DeepseekV3ForCausalLM, DeepSeekV3PretrainedModel, DeepSeekV3Model,
    DeepSeekV3DecoderLayer, DeepseekV3MLAAttention, DeepSeekV3MLP, DeepSeekV3MoE
)
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear, MergedReplicatedLinear, RowParallelLinear, KVBatchLinear
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
from fastdeploy.model_executor.models.model_base import (
    ModelForCasualLM, ModelCategory, ModelRegistry
)
from fastdeploy.model_executor.utils import (
    default_weight_loader, process_weights_after_loading, slice_fn, get_tensor
)

# --- 自定义层 ---
# Kimi的 qkv_a_proj 与 DeepSeekV2 不同，它需要一个能处理独立权重拼接的 loader。
# MergedReplicatedLinear 默认的 loader 不适用，所以我们重写它。
class KimiK2QKVAPool(MergedReplicatedLinear):
    def weight_loader(self, param, loaded_weight, loaded_shard_id: str | None = None):
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        loaded_weight = get_tensor(loaded_weight)
        if weight_need_transpose:
            loaded_weight = loaded_weight.transpose([1, 0])

        assert loaded_shard_id in ["q_a", "kv_a"], f"Invalid shard_id '{loaded_shard_id}' for KimiK2QKVAPool"
        if not param._is_initialized():
            param.initialize()

        output_dim = getattr(param, "output_dim", False)
        
        if loaded_shard_id == "q_a":
            param_shard_offset, param_shard_size = 0, self.output_sizes[0]
        else: # "kv_a"
            param_shard_offset, param_shard_size = self.output_sizes[0], self.output_sizes[1]
        
        param_slice = slice_fn(param, output_dim, start=param_shard_offset, end=param_shard_offset + param_shard_size)
        assert param_slice.shape == loaded_weight.shape, f"Shape mismatch for {self.prefix} shard {loaded_shard_id}"
        param_slice.copy_(loaded_weight, False)

# --- 模型结构定义 ---
class KimiK2MLAAttention(DeepseekV3MLAAttention):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super(DeepseekV3MLAAttention, self).__init__()
        # 强制类型转换，确保健壮性
        self.tp_size = int(fd_config.parallel_config.tensor_parallel_size)
        self.hidden_size = int(fd_config.model_config.hidden_size)
        self.num_attention_heads = int(fd_config.model_config.num_attention_heads)
        self.num_attention_heads_tp = self.num_attention_heads // self.tp_size
        self.v_head_dim = int(fd_config.model_config.v_head_dim)
        self.q_lora_rank = int(fd_config.model_config.q_lora_rank)
        self.kv_lora_rank = int(fd_config.model_config.kv_lora_rank)
        self.qk_nope_head_dim = int(fd_config.model_config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(fd_config.model_config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.rope_theta = float(fd_config.model_config.rope_theta)
        self.rms_norm_eps = float(fd_config.model_config.rms_norm_eps)

        # 使用我们自定义的 KimiK2QKVAPool
        self.qkv_a_proj_with_mqa = KimiK2QKVAPool(fd_config, f"{prefix}.qkv_a_proj_with_mqa", self.hidden_size, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], False)
        
        # 剩余部分与 DeepSeekV3 相同
        self.q_a_layernorm = RMSNorm(fd_config, self.q_lora_rank, self.rms_norm_eps, f"{prefix}.q_a_layernorm")
        self.q_b_proj = ColumnParallelLinear(fd_config, f"{prefix}.q_b_proj", self.q_lora_rank, self.num_attention_heads * self.qk_head_dim, False)
        self.kv_a_layernorm = RMSNorm(fd_config, self.kv_lora_rank, self.rms_norm_eps, f"{prefix}.kv_a_layernorm")
        self.kv_b_proj = ColumnParallelLinear(fd_config, f"{prefix}.kv_b_proj", self.kv_lora_rank, self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim), False)
        self.o_proj = RowParallelLinear(fd_config, f"{prefix}.o_proj", self.num_attention_heads * self.v_head_dim, self.hidden_size, False)
        self.kv_b_proj_bmm = KVBatchLinear(fd_config, self.kv_b_proj, f"{prefix}.kv_b_proj", self.kv_lora_rank, self.num_attention_heads, self.qk_nope_head_dim, self.v_head_dim)
        
        self.rope_scaling = fd_config.model_config.rope_scaling
        self.attn_softmax_scale = self.qk_head_dim**-0.5
        if self.rope_scaling:
            mscale = self.yarn_get_mscale(self.rope_scaling["factor"], float(self.rope_scaling.get("mscale_all_dim", False)))
            self.attn_softmax_scale *= mscale * mscale
        rope_scaling_kwargs = { k: self.rope_scaling[k] for k in ["beta_fast", "beta_slow", "mscale", "mscale_all_dim"] if k in self.rope_scaling }
        self.rotary_emb = DeepseekScalingRotaryEmbedding(self.qk_rope_head_dim, self.rope_scaling["original_max_position_embeddings"], self.rope_theta, self.rope_scaling["factor"], **rope_scaling_kwargs)
        self.mla_attn = Attention(fd_config=fd_config, layer_id=layer_id, prefix=prefix, use_neox_rotary_style=False)
        self.prefix = prefix
    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1: return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

class KimiK2DecoderLayer(DeepSeekV3DecoderLayer):
    def __init__(self, fd_config: FDConfig, prefix: str = ""):
        super(DeepSeekV3DecoderLayer, self).__init__()
        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = KimiK2MLAAttention(fd_config, layer_id, f"{prefix}.self_attn")
        if fd_config.model_config.n_routed_experts and layer_id >= fd_config.model_config.first_k_dense_replace:
            self.mlp = DeepSeekV3MoE(fd_config, layer_id, f"{prefix}.mlp")
        else:
            self.mlp = DeepSeekV3MLP(fd_config, fd_config.model_config.intermediate_size, f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(fd_config, fd_config.model_config.hidden_size, fd_config.model_config.rms_norm_eps, f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, fd_config.model_config.hidden_size, fd_config.model_config.rms_norm_eps, f"{prefix}.post_attention_layernorm")

class KimiK2Model(DeepSeekV3Model):
    def __init__(self, fd_config: FDConfig):
        super(DeepSeekV3Model, self).__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(fd_config, fd_config.model_config.vocab_size, fd_config.model_config.hidden_size, paddle.get_default_dtype(), "model.embed_tokens")
        self.norm = RMSNorm(fd_config, fd_config.model_config.hidden_size, fd_config.model_config.rms_norm_eps, "model.norm")
        self.layers = nn.LayerList([KimiK2DecoderLayer(fd_config, f"model.layers.{i}") for i in range(self.num_layers)])

@ModelRegistry.register_model_class(
    architecture="KimiK2ForCausalLM",
    module_name="kimi_k2",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION
)
class KimiK2ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, fd_config: FDConfig):
        ModelForCasualLM.__init__(self, fd_config)
        self.model = KimiK2Model(fd_config)
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.lm_head = ParallelLMHead(fd_config, fd_config.model_config.hidden_size, fd_config.model_config.vocab_size, "lm_head")
        self.position_ids_buffer = paddle.empty([fd_config.scheduler_config.max_num_batched_tokens], dtype="int32")
        self.mask_encoder_batch_buffer = paddle.empty([fd_config.scheduler_config.max_num_batched_tokens, 1], dtype="int32")

    @classmethod
    def name(cls): return "KimiK2ForCausalLM"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        stacked_params_mapping = [("up_gate_proj", "gate_proj", "gate"), ("up_gate_proj", "up_proj", "up"), ("embed_tokens.embeddings", "embed_tokens", None), ("lm_head.linear", "lm_head", None), ("experts.gate_correction_bias", "gate.e_score_correction_bias", None), ("qkv_a_proj_with_mqa", "q_a_proj", "q_a"), ("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", "kv_a")]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(self.fd_config.model_config.n_routed_experts, "gate_proj", "down_proj", "up_proj", "experts.up_gate_proj_", "experts.down_proj_")
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))

        for loaded_weight_name, loaded_weight in weights_iterator:
            # Kimi权重名不包含 'deepseek_v3'，直接使用
            model_param_name, param = None, None
            for p_name, w_name, s_id in stacked_params_mapping:
                if w_name not in loaded_weight_name or "mlp.experts." in loaded_weight_name: continue
                model_param_name = loaded_weight_name.replace(w_name, p_name)
                if model_param_name not in params_dict: continue
                param = params_dict[model_param_name]
                getattr(param, "weight_loader", default_weight_loader(self.fd_config))(param, loaded_weight, s_id)
                break
            else:
                for p_name, w_name, e_id, s_id in expert_params_mapping:
                    if w_name not in loaded_weight_name: continue
                    model_param_name = loaded_weight_name.replace(w_name, p_name)
                    if model_param_name not in params_dict: continue
                    param = params_dict[model_param_name]
                    param.weight_loader(param, loaded_weight, shard_id=s_id, expert_id=e_id)
                    break
                else:
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict: continue
                    param = params_dict[model_param_name]
                    getattr(param, "weight_loader", default_weight_loader(self.fd_config))(param, loaded_weight)
            
            if model_param_name is None: model_param_name = loaded_weight_name
            if param is None: param = params_dict.get(model_param_name)
            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            if "kv_b_proj" in model_sublayer_name:
                process_weights_after_loading_fn(model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm"))
            process_weights_after_loading_fn(model_sublayer_name, param)

class KimiK2PretrainedModel(DeepSeekV3PretrainedModel):
    @classmethod
    def arch_name(cls): return "KimiK2ForCausalLM"