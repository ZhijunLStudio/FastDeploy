# fastdeploy/model_executor/models/minimax_m1.py (Final, Cleaned Version without _get_output)

from __future__ import annotations
import math
import re
from typing import Optional
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
    ReplicatedLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.utils import default_weight_loader
from fastdeploy.model_executor.layers.utils import get_tensor

# NO _get_output HELPER FUNCTION

class MiniMaxM1MLP(nn.Layer):
    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = "", reduce_results: bool = True):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(fd_config, prefix=f"{prefix}.up_gate_proj", input_size=config.hidden_size, output_size=intermediate_size * 2, with_bias=False)
        self.down_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.down_proj", input_size=intermediate_size, output_size=config.hidden_size, with_bias=False, reduce_results=reduce_results)
        self.act_fn = SiluAndMul()
    def forward(self, x):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

class MiniMaxM1MoEBlock(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear(fd_config, prefix=f"{prefix}.gate", input_size=config.hidden_size, output_size=config.num_local_experts, with_bias=False, weight_dtype="float32")
        self.experts = FusedMoE(fd_config, moe_intermediate_size=config.intermediate_size, num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, layer_idx=layer_id)
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        router_logits = self.gate(hidden_states.cast("float32"))
        return self.experts(hidden_states, router_logits)

class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, original_layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.original_layer_id = original_layer_id
        attn_prefix = f"{prefix}.self_attn"
        self.attn_type = 1 
        logger.info(f"Initializing DecoderLayer with prefix '{prefix}' (maps to original layer {original_layer_id}) as GQA type.")
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{attn_prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(fd_config, prefix=f"{attn_prefix}.o_proj", input_size=config.num_attention_heads * config.head_dim, output_size=config.hidden_size, with_bias=False)
        self.self_attn = Attention(fd_config, layer_id=original_layer_id, prefix=attn_prefix, use_neox_rotary_style=True)
        self.mlp = MiniMaxM1MoEBlock(fd_config, original_layer_id, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.post_attention_layernorm")

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: Optional[paddle.Tensor]):
        if residual is None:
            residual = hidden_states
            normed_hidden_states = self.input_layernorm(hidden_states)
        else:
            normed_hidden_states, residual = self.input_layernorm(hidden_states, residual)

        qkv_out = self.qkv_proj(normed_hidden_states)
        attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
        attn_output = self.o_proj(attn_output)
        
        normed_attn_output, residual = self.post_attention_layernorm(attn_output, residual)
        
        mlp_output = self.mlp(normed_attn_output)
        
        return mlp_output, residual

@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(fd_config, num_embeddings=self.config.vocab_size, embedding_dim=self.config.hidden_size, prefix=f"{prefix}.embed_tokens")
        layer_mapping = getattr(self.config, "layer_mapping", range(self.config.num_hidden_layers))
        if len(layer_mapping) != self.config.num_hidden_layers:
            raise ValueError(f"Length of `layer_mapping` ({len(layer_mapping)}) must match `num_hidden_layers` ({self.config.num_hidden_layers}).")
        self.layers = nn.LayerList([MiniMaxM1DecoderLayer(fd_config, original_layer_id=layer_mapping[i], prefix=f"{prefix}.layers.{layer_mapping[i]}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(fd_config, hidden_size=self.config.hidden_size, eps=self.config.rms_norm_eps, prefix=f"{prefix}.norm")

    def forward(self, *args, **kwargs):
        ids_remove_padding = kwargs.get("ids_remove_padding")
        forward_meta = kwargs.get("forward_meta")

        # Direct call, no _get_output
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=residual)
        
        if residual is not None:
             hidden_states = hidden_states + residual
        
        # Direct call, no _get_output
        out = self.norm(hidden_states)
        
        return out

class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        config = self.fd_config.model_config
        config.pretrained_config.prefix_name = "model"
        if hasattr(config, "num_local_experts") and not hasattr(config, "moe_num_experts"):
            config.moe_num_experts = config.num_local_experts
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim") and config.rotary_dim < config.head_dim:
            config.partial_rotary_factor = config.rotary_dim / config.head_dim
        if not hasattr(config, "first_k_dense_replace"):
            config.first_k_dense_replace = 0
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, embedding_dim=config.hidden_size, num_embeddings=config.vocab_size, prefix="lm_head")
    
    @classmethod
    def name(cls): return "MiniMaxM1ForCausalLM"
    
    def forward(self, *args, **kwargs): return self.model(**kwargs)
    
    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs): return self.lm_head(hidden_states)

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        # (The robust weight loader remains the same, it was correct)
        logger.info("Initializing robust multi-GPU weight loader for MiniMax-M1...")
        params_dict = dict(self.named_parameters())
        config = self.fd_config.model_config
        layer_mapping = getattr(config, "layer_mapping", range(config.num_hidden_layers))
        tp_rank = self.fd_config.parallel_config.tensor_parallel_rank
        tp_size = self.fd_config.parallel_config.tensor_parallel_size
        is_torch_format = config.model_format == "torch"
        def manual_moe_loader(param, loaded_weight, expert_id, shard_id):
            if not param._is_initialized(): param.set_value(paddle.zeros(param.shape, dtype=param.dtype))
            loaded_weight_tensor = get_tensor(loaded_weight)
            if is_torch_format: loaded_weight_tensor = loaded_weight_tensor.transpose([1, 0])
            if shard_id in ["gate", "up"]:
                output_size_per_shard = loaded_weight_tensor.shape[1] // tp_size
                start, end = tp_rank * output_size_per_shard, (tp_rank + 1) * output_size_per_shard
                loaded_weight_shard = loaded_weight_tensor[:, start:end]
                target_expert_slice = param[expert_id]
                intermediate_size_sharded = target_expert_slice.shape[1] // 2
                if shard_id == "gate": target_sub_slice = target_expert_slice[:, :intermediate_size_sharded]
                else: target_sub_slice = target_expert_slice[:, intermediate_size_sharded:]
                if target_sub_slice.shape != loaded_weight_shard.shape: raise ValueError(f"[MoE Shape Mismatch] gate/up exp {expert_id}: Param={target_sub_slice.shape}, Loaded={loaded_weight_shard.shape}")
                target_sub_slice.set_value(loaded_weight_shard)
            else:
                input_size_per_shard = loaded_weight_tensor.shape[0] // tp_size
                start, end = tp_rank * input_size_per_shard, (tp_rank + 1) * input_size_per_shard
                loaded_weight_shard = loaded_weight_tensor[start:end, :]
                target_expert_slice = param[expert_id]
                if target_expert_slice.shape != loaded_weight_shard.shape: raise ValueError(f"[MoE Shape Mismatch] down exp {expert_id}: Param={target_expert_slice.shape}, Loaded={loaded_weight_shard.shape}")
                target_expert_slice.set_value(loaded_weight_shard)
        stacked_params_mapping = [
            ("qkv_proj.weight", "q_proj.weight", "q"),("qkv_proj.weight", "k_proj.weight", "k"),("qkv_proj.weight", "v_proj.weight", "v"),
            ("shared_mlp.up_gate_proj.weight", "gate_proj.weight", 0),("shared_mlp.up_gate_proj.weight", "up_proj.weight", 1),
            ("model.embed_tokens.embeddings.weight", "model.embed_tokens.weight", None),("lm_head.linear.weight", "lm_head.weight", None),
        ]
        for loaded_weight_name, loaded_weight in weights_iterator:
            found = False
            moe_match = re.search(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
            if moe_match:
                original_layer_idx, expert_id, weight_type = int(moe_match.group(1)), int(moe_match.group(2)), moe_match.group(3)
                if original_layer_idx in layer_mapping:
                    layer_idx_in_model = layer_mapping.index(original_layer_idx)
                    if weight_type in ["w1", "w3"]:
                        param_name = f"model.layers.{layer_idx_in_model}.mlp.experts.up_gate_proj_weight"
                        shard_id = "gate" if weight_type == "w1" else "up"
                    else:
                        param_name = f"model.layers.{layer_idx_in_model}.mlp.experts.down_proj_weight"
                        shard_id = "down"
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        manual_moe_loader(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
                        found = True
            if found: continue
            for fd_part, hf_part, shard_id in stacked_params_mapping:
                if hf_part in loaded_weight_name:
                    fd_name = loaded_weight_name.replace(hf_part, fd_part)
                    if fd_name in params_dict:
                        param = params_dict[fd_name]
                        loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                        loader(param, loaded_weight, shard_id)
                        found = True
                        break
            if found: continue
            fd_name = loaded_weight_name.replace("block_sparse_moe.gate", "mlp.gate")
            if fd_name in params_dict:
                param = params_dict[fd_name]
                loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                loader(param, loaded_weight)
                found = True
        logger.info("Finished processing weights.")

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method.")

class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig
    @classmethod
    def arch_name(cls): return "MiniMaxM1ForCausalLM"
    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.info("Bypassing automatic tensor parallel mappings for MiniMax-M1.")
        return {}